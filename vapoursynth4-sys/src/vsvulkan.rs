/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

//! The GPU side of the API, from `VSVulkan4.h`.
//!
//! Obtained through [`getVulkanAPI()`](crate::VSAPI::getVulkanAPI). It is
//! versioned by the core API rather than negotiated, so it has no version of its
//! own and the call cannot fail.
//!
//! The shape of it is deliberately raw: the core hands out its Vulkan handles
//! and per-plane buffers, and a filter brings its own pipelines, command buffers
//! and synchronization on top of them. Vulkan 1.4 is required rather than
//! negotiated. The Vulkan types live in [`crate::vulkan`].
//!
//! The contract in short:
//!
//! - Resolve every Vulkan entry point through
//!   [`get_instance_proc_addr`](VSVulkanCoreHandles::get_instance_proc_addr) or
//!   `vkGetDeviceProcAddr` from the handles. Nothing is linked.
//! - `vkQueueSubmit` on the shared queues must happen with the matching queue
//!   lock held; see [`lockVulkanQueue()`](VSVULKANAPI::lockVulkanQueue).
//! - Before reading a plane on the GPU, wait for its
//!   [`ready_semaphore`](VSVulkanPlaneInfo::ready_semaphore) and
//!   [`ready_value`](VSVulkanPlaneInfo::ready_value) pair. After producing one,
//!   publish your own pair through
//!   [`setGPUPlaneProducer()`](VSVULKANAPI::setGPUPlaneProducer). A null
//!   semaphore means host-produced content that is ready immediately.
//! - Hold a reference to every frame a submission touches until that submission
//!   has completed.
//! - GPU producing filters declare [`VSFilterFlags::GPUOutput`] and create their
//!   nodes with [`createVideoFilterEx()`](crate::VSAPI::createVideoFilterEx) or
//!   [`createVideoFilterEx2()`](crate::VSAPI::createVideoFilterEx2).

// The Vulkan module is deliberately taken wholesale: it is one generated unit
// whose type set is decided by `VSVulkan4.h`, not by anything this file names.
#![allow(clippy::wildcard_imports)]

use core::ffi::{c_char, c_int, c_void};

use super::vulkan::*;
use super::{VSFrame, VSFrameContext, VSNode, VSCore, VSVideoFormat, opaque_struct};

/// Whether a wait left it safe to destroy what a submission might still have
/// been using.
///
/// This is written as the positive condition on purpose, so that a value added
/// to [`VSGPUDrainResult`] later is unsafe by default. Written as
/// `result != Incomplete` instead, a new value would silently land on the safe
/// side and start destroying pipelines and buffers under a live dispatch.
#[must_use]
pub const fn gpu_drain_safe_to_destroy(drain_result: VSGPUDrainResult) -> bool {
    matches!(
        drain_result,
        VSGPUDrainResult::Drained | VSGPUDrainResult::DeviceLost
    )
}

/// Which handle a Vulkan function is resolved against.
///
/// Device level entry points are deliberately fetched with `vkGetDeviceProcAddr`:
/// resolving them through the instance yields a loader trampoline that has to
/// dispatch on the first argument of every single call.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum VSVulkanLevel {
    /// Resolved from a null instance, before one exists.
    Global = 0,
    Instance = 1,
    Device = 2,
}

/// Whether a Vulkan entry point may be absent.
///
/// Optional entries are allowed to stay null and must be null checked at the
/// call site. Everything else is core Vulkan 1.4, and its absence means the
/// device lied about its version, which is fatal.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum VSVulkanRequirement {
    Required = 0,
    Optional = 1,
}

/// Which of the core's queues a submission goes on.
///
/// The two are *not* interchangeable, and Vulkan guarantees the implication one
/// way only: a compute queue always accepts transfer commands, while a dedicated
/// transfer queue need not accept compute ones. A discrete card's DMA family
/// typically reports `VK_QUEUE_TRANSFER_BIT` alone. So
/// [`Compute`](VSVulkanQueueType::Compute) takes anything, and
/// [`Transfer`](VSVulkanQueueType::Transfer) must be given copies only.
///
/// Recording a dispatch against [`Transfer`](VSVulkanQueueType::Transfer) is
/// invalid usage wherever a real DMA family exists, and silently fine on the
/// hardware where it resolves to a compute queue, which is what makes the
/// mistake easy to ship. When in doubt use
/// [`Compute`](VSVulkanQueueType::Compute): the cost is losing overlap with the
/// core's own transfers, not correctness.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum VSVulkanQueueType {
    /// The compute family's queue. Accepts anything.
    Compute = 0,
    /// The transfer family's queue; without a transfer family the compute
    /// family's second queue where it has one, and only failing both the same
    /// underlying queue as [`Compute`](VSVulkanQueueType::Compute).
    ///
    /// The core's uploads always run on this queue, and its frame downloads too
    /// unless the transfer family has a second queue, which is never handed out.
    Transfer = 1,
}

/// What a host wait established.
///
/// Returned by all three wait functions --
/// [`gpuExecPoolWaitIdle()`](VSVULKANAPI::gpuExecPoolWaitIdle),
/// [`gpuExecWaitValue()`](VSVULKANAPI::gpuExecWaitValue) and
/// [`gpuTimelineWaitValue()`](VSVULKANAPI::gpuTimelineWaitValue) -- because the
/// question a caller asks after a wait is the same one every time. Anything
/// other than [`Drained`](VSGPUDrainResult::Drained) is a failure for a caller
/// that only wants to know whether it can carry on.
///
/// The distinction matters to anything deciding whether resources a submission
/// might still be using are safe to destroy; see
/// [`gpu_drain_safe_to_destroy()`].
///
/// [`Drained`](VSGPUDrainResult::Drained) is as strong as the driver allows,
/// which is not absolute. A driver that answers a reset by signalling exactly
/// the values the abandoned submissions were going to signal defeats it, and no
/// Vulkan query separates that from real completion. A filter that must be
/// certain of a result has to check the result itself.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum VSGPUDrainResult {
    /// The wait established completion, and for a pool drain every release
    /// callback has run.
    Drained = 0,
    /// The wait could not be established, so submissions may still be running.
    Incomplete = 1,
    /// The device was reset or lost, so nothing is executing any more.
    DeviceLost = 2,
}

/// A language understood by
/// [`compileGPUShader()`](VSVULKANAPI::compileGPUShader).
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum VSGPUShaderLanguage {
    /// Compute stage GLSL. The core pins the accepted dialect as a platform
    /// property: write `#version 460`, compiled for the Vulkan 1.4 client
    /// targeting SPIR-V 1.6.
    GLSL = 0,
}

/// Cleanup a filter hands to an exec context, run once the submission it was
/// recorded against has completed.
///
/// See [`gpuExecRetain()`](VSVULKANAPI::gpuExecRetain).
pub type VSGPUReleaseFunc = Option<unsafe extern "system-unwind" fn(object: *mut c_void)>;

/// Whether holding an import of memory [`exportGPUPlane()`] handed out costs the
/// process that memory a second time, on top of the core's own allocation.
///
/// `1` where it does, so an importer declares its imports with
/// [`reserveGPUMemory()`], and `0` where it does not, so declaring them would
/// only make the core cache fewer frames.
///
/// Windows charges a shared allocation against the process's budget again for
/// every device that opens it. On Linux an import is the same kernel buffer
/// object as the core's allocation and takes no memory of its own.
///
/// [`exportGPUPlane()`]: VSVULKANAPI::exportGPUPlane
/// [`reserveGPUMemory()`]: VSVULKANAPI::reserveGPUMemory
#[cfg(windows)]
pub const VS_GPU_IMPORT_CHARGED_AGAIN: c_int = 1;
/// See the Windows variant: an import here shares the core's allocation.
#[cfg(not(windows))]
pub const VS_GPU_IMPORT_CHARGED_AGAIN: c_int = 0;

opaque_struct!(
    /// A timeline semaphore for publishing producer pairs, reference counted and
    /// owned by the core.
    ///
    /// A filter signals it and hands it to
    /// [`setGPUPlaneProducer()`](VSVULKANAPI::setGPUPlaneProducer). Every plane
    /// it is published on keeps a reference of its own until the plane is freed,
    /// so the semaphore lives for as long as something might still wait on it.
    ///
    /// A filter's timeline therefore need not outlive its consumers: release your
    /// reference once you are done signalling, in the free callback, and any
    /// frame still in flight keeps the semaphore alive. Frames legitimately
    /// outlive the filter that made them -- `std.FrameEval` and
    /// `std.ModifyFrame` return one from a node they then drop, and the cache
    /// can hold one indefinitely -- which is what the counting is for.
    VSGPUTimeline,
    /// A scratch buffer from the core's pooled VRAM allocator, for filters
    /// needing memory that is not a frame plane: reduction partials, lookup
    /// tables, intermediate rows.
    ///
    /// Owned through the opaque handle; [`VSVulkanBufferInfo`] is everything a
    /// kernel or the host needs to use it.
    VSGPUBuffer,
    /// A bare region of the same pooled VRAM, for resources the core has no
    /// constructor for.
    ///
    /// An image is the case that motivates it: memory is never passed to
    /// `vkCreateImage`, it is bound afterwards, so nothing is gained by wrapping
    /// image creation. The filter creates the image its own way and only the
    /// allocation comes from here.
    VSGPUMemory,
    /// A declaration of GPU memory allocated outside the core -- by CUDA,
    /// another Vulkan device ecosystem, a video session -- so the byte count
    /// takes part in the core's budgeting.
    ///
    /// Holds no memory itself, only the number. See
    /// [`reserveGPUMemory()`](VSVULKANAPI::reserveGPUMemory).
    VSGPUMemoryReservation,
    /// An exec pool, owning a timeline semaphore, a command pool and between two
    /// and eight command buffers.
    ///
    /// A pool belongs to one filter instance. See
    /// [`createGPUExecPool()`](VSVULKANAPI::createGPUExecPool).
    VSGPUExecPool,
    /// One recording of a [`VSGPUExecPool`], claimed by one thread from
    /// [`gpuExecAcquire()`](VSVULKANAPI::gpuExecAcquire) until
    /// [`gpuExecSubmit()`](VSVULKANAPI::gpuExecSubmit) or
    /// [`gpuExecAbandon()`](VSVULKANAPI::gpuExecAbandon).
    VSGPUExecContext,
    /// A runtime compiled shader as an opaque handle holding the SPIR-V words.
    ///
    /// Independent of everything else once returned: it stays valid after the
    /// core that compiled it is freed, and is released with
    /// [`freeGPUShader()`](VSVULKANAPI::freeGPUShader).
    VSGPUShader
);

/// Everything needed to run your own Vulkan work on the core's device.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VSVulkanCoreHandles {
    pub instance: VkInstance,
    pub physical_device: VkPhysicalDevice,
    pub device: VkDevice,
    /// For anything outside the curated set
    /// [`getVulkanFunctions()`](VSVULKANAPI::getVulkanFunctions) returns.
    pub get_instance_proc_addr: PFN_vkGetInstanceProcAddr,
    pub compute_queue_family: u32,
    pub compute_queue_index: u32,
    /// With [`transfer_queue_index`](Self::transfer_queue_index), the compute
    /// family's second queue when there is no transfer family but the compute
    /// family has two queues.
    ///
    /// Equal to the compute values when it has only one.
    pub transfer_queue_family: u32,
    pub transfer_queue_index: u32,
}

/// One GPU resident plane: a linear pitched storage buffer laid out exactly like
/// the equivalent CPU plane, so [`getStride()`](crate::VSAPI::getStride) and the
/// frame dimension functions apply unchanged.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VSVulkanPlaneInfo {
    pub buffer: VkBuffer,
    /// `stride * height` bytes.
    pub buffer_size: VkDeviceSize,
    /// Wait `(ready_semaphore, ready_value)` before reading; `None` means ready
    /// now.
    pub ready_semaphore: VkSemaphore,
    pub ready_value: u64,
}

/// What a kernel or the host needs to use a [`VSGPUBuffer`].
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VSVulkanBufferInfo {
    pub buffer: VkBuffer,
    /// Nonzero when the usage included `SHADER_DEVICE_ADDRESS`.
    pub address: VkDeviceAddress,
    /// Nonnull when the memory ended up host visible; persistently mapped.
    pub mapped: *mut c_void,
    /// The requested size, unrounded.
    pub size: VkDeviceSize,
    /// What the chosen memory type actually provides.
    pub memory_flags: VkMemoryPropertyFlags,
}

/// What a kernel or the host needs to use a [`VSGPUMemory`] region.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VSVulkanMemoryInfo {
    /// The shared block; bind against it, never free it.
    pub memory: VkDeviceMemory,
    /// Where the region starts in that block; pass it to `vkBind*Memory2`.
    pub offset: VkDeviceSize,
    /// What the region actually reserves, at least the requested size.
    pub size: VkDeviceSize,
    /// Nonnull when host visible, already offset to the region.
    pub mapped: *mut c_void,
    /// What the chosen memory type actually provides.
    pub memory_flags: VkMemoryPropertyFlags,
}

/// A frame plane's backing memory exported as an opaque handle, so CUDA or
/// another Vulkan device in the process can wrap the allocation and read or
/// write the plane zero copy.
///
/// The plane lives at [`offset`](Self::offset) within an allocation of
/// [`memory_size`](Self::memory_size) bytes from memory type
/// [`memory_type_index`](Self::memory_type_index). Import the whole allocation
/// once and address planes by offset.
///
/// [`memory_id`](Self::memory_id) is stable for the allocation's lifetime and
/// never reused, while every export call returns a *new* handle, so key cached
/// imports by [`memory_id`](Self::memory_id) and never by handle value, and
/// close surplus handles.
///
/// # Safety
///
/// The handle belongs to the caller: an NT handle is not consumed by Vulkan or
/// CUDA import, so `CloseHandle` it once the import exists; a POSIX fd is
/// consumed by a successful import and must only be closed when the import
/// failed.
///
/// The OS reference-counts the memory, so a cached import may outlive the
/// frames, and the core, that led to it. Synchronization is host side: call
/// [`waitGPUFrame()`](VSVULKANAPI::waitGPUFrame) before reading through an
/// import -- a bare producer pair wait is not enough -- and finish foreign
/// writes before returning a frame containing them.
///
/// Input frames are handed over by taking them with
/// [`getExportableFrameFilter()`](VSVULKANAPI::getExportableFrameFilter), for
/// reading only, and never come back. Exporting any other plane fails.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VSVulkanExportedMemory {
    pub memory_id: u64,
    pub memory_size: VkDeviceSize,
    /// The memory type the allocation was made from.
    ///
    /// An opaque handle carries no memory type information of its own: Vulkan
    /// forbids asking which types it accepts, and requires it to be imported
    /// with exactly the exporting allocation's `memoryTypeIndex` and
    /// `allocationSize`. A Vulkan importer therefore passes this and
    /// [`memory_size`](Self::memory_size) to `vkAllocateMemory` unchanged, which
    /// is well defined because type numbering is per physical device and sharing
    /// is only defined with a core on the same one. CUDA importers have no use
    /// for it.
    pub memory_type_index: u32,
    pub offset: VkDeviceSize,
    /// The plane's bytes, `stride * height`.
    pub size: VkDeviceSize,
    /// The `VkExternalMemoryHandleTypeFlagBits` of the handle.
    pub handle_type: c_int,
    /// `HANDLE` on Windows, file descriptor elsewhere.
    pub handle: isize,
}

/// A timeline semaphore exported as an opaque handle, from
/// [`exportGPUSemaphore()`](VSVULKANAPI::exportGPUSemaphore).
///
/// Importing a producer pair's semaphore lets a CUDA stream or another Vulkan
/// device wait the pair on the device, replacing
/// [`waitGPUFrame()`](VSVULKANAPI::waitGPUFrame)'s host wait and restoring full
/// pipelining across the API boundary. An external semaphore wait also carries
/// the cross device memory dependency, so no separate availability flush is
/// needed on that path.
///
/// Every call returns a *new* handle for the same semaphore. Cache imports keyed
/// by the `VkSemaphore` value from [`VSVulkanPlaneInfo`], scoped to your filter
/// instance: a plane holds a reference to the timeline it names, so as long as
/// you hold the frame the handle stays live and the key cannot go stale.
///
/// # Safety
///
/// Handle ownership follows the same platform rules as memory export: close NT
/// handles after importing, and note that fds are consumed by a successful
/// import.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VSVulkanExportedSemaphore {
    /// The `VkExternalSemaphoreHandleTypeFlagBits` of the handle.
    pub handle_type: c_int,
    /// `HANDLE` on Windows, file descriptor elsewhere.
    pub handle: isize,
}

/// What the core ended up with and where its VRAM budget stands.
///
/// Returned by [`getVulkanCoreInfo()`](VSVULKANAPI::getVulkanCoreInfo).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VSVulkanCoreInfo {
    pub device_name: [c_char; 256],
    /// Largest device local heap.
    pub device_memory: i64,
    /// What the driver says this process may reasonably use right now.
    pub budget: i64,
    /// Current `VapourSynth` use of device memory: frames, pooled buffers,
    /// reservations and any transfer staging the driver placed there.
    ///
    /// Staging in system RAM, the discrete card case, counts toward
    /// [`VSCoreInfo::used_framebuffer_size`](crate::VSCoreInfo::used_framebuffer_size)
    /// instead.
    pub allocated: i64,
    /// The eviction limit, settable through
    /// [`setMaxVRAMUse()`](VSVULKANAPI::setMaxVRAMUse).
    pub limit: i64,
    /// Identity for matching against other APIs' device enumerations.
    ///
    /// CUDA exposes the same UUID. The LUID pairs with DXGI adapters and is only
    /// meaningful when [`device_luid_valid`](Self::device_luid_valid).
    pub device_uuid: [u8; VK_UUID_SIZE],
    pub device_luid: [u8; VK_LUID_SIZE],
    pub device_node_mask: u32,
    pub device_luid_valid: c_int,
    /// The `VkExternalMemoryHandleTypeFlagBits`
    /// [`exportGPUPlane()`](VSVULKANAPI::exportGPUPlane) hands out, or `0` when
    /// export is unavailable.
    pub export_handle_type: c_int,
    /// The `VkExternalSemaphoreHandleTypeFlagBits`
    /// [`exportGPUSemaphore()`](VSVULKANAPI::exportGPUSemaphore) hands out, or
    /// `0` when unavailable.
    pub semaphore_export_handle_type: c_int,
    /// Nonzero when the device's memory is the host's, which is the case for
    /// integrated and software devices.
    ///
    /// [`device_memory`](Self::device_memory) and [`budget`](Self::budget) then
    /// describe a share of system RAM rather than separate hardware, and
    /// [`limit`](Self::limit) is capped so that it and the host memory limit
    /// together leave the machine some room. A plugin sizing its own pools should
    /// do likewise.
    pub unified_memory: c_int,
}

/// One entry per physical device; the position in the enumeration is exactly the
/// index [`setVulkanDevice()`](VSVULKANAPI::setVulkanDevice) takes.
///
/// Unusable devices are listed too and may be selected, which then fails with
/// their reason, so a frontend can present everything and explain refusals.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VSVulkanDeviceListEntry {
    pub device_name: [c_char; 256],
    pub api_version: u32,
    /// A `VkPhysicalDeviceType`.
    pub device_type: c_int,
    /// Largest device local heap.
    pub device_memory: i64,
    /// Passes the Vulkan 1.4 and required feature gate.
    pub usable: c_int,
    pub unusable_reason: [c_char; 256],
    pub device_uuid: [u8; VK_UUID_SIZE],
    pub device_luid: [u8; VK_LUID_SIZE],
    pub device_node_mask: u32,
    pub device_luid_valid: c_int,
}

// MARK: VSVULKANAPI

/// The GPU side of the API, retrieved through
/// [`getVulkanAPI()`](crate::VSAPI::getVulkanAPI).
///
/// Like [`VSAPI`](crate::VSAPI), this is a run of function pointers read by
/// offset. A new entry point goes with the ones it belongs to, which moves every
/// pointer after it, so a plugin left over from an older header still links and
/// still runs, and simply calls the neighbouring function. It is versioned by
/// the core API rather than negotiated:
/// [`getAPIVersion()`](crate::VSAPI::getAPIVersion) answers which one you got.
///
/// Unless an entry says otherwise, a status-returning function returns `0` on
/// success and nonzero on failure, and one returning a handle returns `NULL` on
/// failure. Either way the reason lands in `errorMessage` where the function
/// takes one.
#[allow(non_snake_case)]
#[repr(C)]
pub struct VSVULKANAPI {
    // MARK: Device selection and information

    /// Lists every physical device through a temporary instance, so it works
    /// before any device selection and needs no core.
    ///
    /// # Arguments
    ///
    /// * `entries` - Where the entries are written, or `NULL` to just count.
    ///
    /// * `maxEntries` - Capacity of `entries`.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// # Return
    ///
    /// The total device count, which may exceed `maxEntries`, or `-1` with the
    /// error set.
    pub enumerateVulkanDevices: unsafe extern "system-unwind" fn(
        entries: *mut VSVulkanDeviceListEntry,
        maxEntries: c_int,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Picks which of the enumerated devices this core runs on.
    ///
    /// # Arguments
    ///
    /// * `core` - The core to configure.
    ///
    /// * `deviceIndex` - A position from the enumeration, or `-1` for the most
    ///   powerful one.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// # Warning
    ///
    /// Only valid before the device is first used.
    pub setVulkanDevice: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        deviceIndex: c_int,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Reports what the core ended up with and where its VRAM budget stands.
    ///
    /// # Arguments
    ///
    /// * `core` - The core to query.
    ///
    /// * `info` - Filled in with the core's device information.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// Brings the device up on first call, like the first GPU filter would.
    pub getVulkanCoreInfo: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        info: *mut VSVulkanCoreInfo,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Sets the VRAM equivalent of [`setMaxCacheSize()`](crate::VSAPI::setMaxCacheSize).
    ///
    /// # Arguments
    ///
    /// * `bytes` - The new limit, in bytes.
    ///
    /// * `core` - The core to configure.
    ///
    /// # Return
    ///
    /// The previous limit.
    pub setMaxVRAMUse:
        unsafe extern "system-unwind" fn(bytes: i64, core: *mut VSCore) -> i64,

    // MARK: Calling Vulkan on the core's device

    /// Reports the instance, physical device, device and queue indices.
    ///
    /// # Arguments
    ///
    /// * `core` - The core to query.
    ///
    /// * `handles` - Filled in with the core's Vulkan handles.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// Brings the device up on first call, like the first GPU filter would.
    pub getVulkanHandles: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        handles: *mut VSVulkanCoreHandles,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Retrieves the core's ready loaded dispatch table, the normal way for
    /// filters to call Vulkan.
    ///
    /// # Arguments
    ///
    /// * `core` - The core to query.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// The handles' [`get_instance_proc_addr`](VSVulkanCoreHandles::get_instance_proc_addr)
    /// stays available for anything outside the curated set. Brings the device
    /// up on first call and stays valid for the core's lifetime.
    pub getVulkanFunctions: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> *const VSVulkanFunctions,
    /// Locks one of the core's shared Vulkan queues.
    ///
    /// # Arguments
    ///
    /// * `core` - The core owning the queue.
    ///
    /// * `queue` - A [`VSVulkanQueueType`].
    ///
    /// # Safety
    ///
    /// Mandatory around every `vkQueueSubmit` you make on the shared queues.
    ///
    /// This is one of the core's own locks, so treat it as a *leaf*: submit
    /// between the two calls and do nothing else. In particular call nothing
    /// from this API inside the bracket. Allocating GPU memory, creating a
    /// frame, acquiring from or waiting on an exec pool, or creating or freeing
    /// one all reach the core's exec registry, which the core's own frame path
    /// locks before this queue, so holding this across one of them deadlocks
    /// against an ordinary cache sweep on a worker thread.
    ///
    /// Do not hold both queues' locks at once either:
    /// [`Transfer`](VSVulkanQueueType::Transfer) and
    /// [`Compute`](VSVulkanQueueType::Compute) are the same non-recursive lock
    /// on a device with neither a dedicated transfer family nor a second compute
    /// queue, so on that hardware the second call never returns.
    pub lockVulkanQueue: unsafe extern "system-unwind" fn(core: *mut VSCore, queue: c_int),
    /// Unlocks a queue locked by [`lockVulkanQueue()`](Self::lockVulkanQueue).
    ///
    /// # Arguments
    ///
    /// * `core` - The core owning the queue.
    ///
    /// * `queue` - A [`VSVulkanQueueType`].
    pub unlockVulkanQueue: unsafe extern "system-unwind" fn(core: *mut VSCore, queue: c_int),

    // MARK: GPU resident frames

    /// Creates a GPU resident video frame for filter output.
    ///
    /// # Arguments
    ///
    /// * `format` - The output format of the frame.
    ///
    /// * `width` - The width of the frame.
    ///
    /// * `height` - The height of the frame.
    ///
    /// * `propSrc` - A frame whose properties are copied to the new frame,
    ///   or `NULL` for none.
    ///
    /// * `core` - The core to create the frame on.
    ///
    /// Identical semantics to
    /// [`newVideoFrame()`](crate::VSAPI::newVideoFrame) otherwise, except that
    /// `NULL` is returned when no Vulkan device is available. Note that
    /// [`newVideoFrame2()`](crate::VSAPI::newVideoFrame2) also supports GPU
    /// resident frames.
    pub newGPUVideoFrame: unsafe extern "system-unwind" fn(
        format: *const VSVideoFormat,
        width: c_int,
        height: c_int,
        propSrc: *const VSFrame,
        core: *mut VSCore,
    ) -> *mut VSFrame,
    /// Retrieves a GPU resident plane of a frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame to query.
    ///
    /// * `plane` - The plane index.
    ///
    /// * `info` - Filled in with the plane's buffer and producer pair.
    ///
    /// # Return
    ///
    /// Nonzero when the frame is not GPU resident or the plane does not exist.
    pub getGPUPlane: unsafe extern "system-unwind" fn(
        frame: *const VSFrame,
        plane: c_int,
        info: *mut VSVulkanPlaneInfo,
    ) -> c_int,
    /// Publishes the producer pair of a plane you write.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame owning the plane.
    ///
    /// * `plane` - The plane index.
    ///
    /// * `timeline` - The timeline the value is signalled on, or `NULL` to
    ///   publish the plane as host ready.
    ///
    /// * `value` - The value on `timeline` that marks the plane as written.
    ///
    /// The plane takes its own reference to `timeline`.
    ///
    /// # Safety
    ///
    /// Fatal on a plane the frame shares with another, for the reason
    /// [`gpuExecWritesPlane()`](Self::gpuExecWritesPlane) gives.
    ///
    /// Publish a value whose signal is already on its way: one the semaphore has
    /// reached, or one that work you have already handed to a queue or a stream
    /// will signal on its own. *Never* a value you still have to signal yourself
    /// from the host. Whoever waits on this pair may be holding locks inside the
    /// core -- freeing a frame waits out its producer, and cache eviction frees
    /// frames -- so a signal that needs a host thread to get around to it can be
    /// waiting on the very thread it is blocking.
    pub setGPUPlaneProducer: unsafe extern "system-unwind" fn(
        frame: *const VSFrame,
        plane: c_int,
        timeline: *mut VSGPUTimeline,
        value: u64,
    ),

    // MARK: Timelines, the semaphores producer pairs are published on

    /// Creates a timeline of your own, for filters recording and submitting
    /// without the core's exec pool.
    ///
    /// # Arguments
    ///
    /// * `core` - The core to create the timeline on.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// It is created with an initial value of 0 and is exportable where the
    /// device allows it. The returned reference is yours to release, in the free
    /// callback and without waiting for consumers, since planes you published it
    /// on hold their own. Returns `NULL` on error.
    pub createGPUTimeline: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> *mut VSGPUTimeline,
    /// Releases one reference to a timeline.
    ///
    /// # Arguments
    ///
    /// * `timeline` - The timeline to release.
    pub freeGPUTimeline: unsafe extern "system-unwind" fn(timeline: *mut VSGPUTimeline),
    /// Takes another reference to a timeline, for handing it to something with
    /// its own lifetime.
    ///
    /// # Arguments
    ///
    /// * `timeline` - The timeline to retain.
    ///
    /// Every added reference needs a matching
    /// [`freeGPUTimeline()`](Self::freeGPUTimeline).
    pub addGPUTimelineRef: unsafe extern "system-unwind" fn(timeline: *mut VSGPUTimeline),
    /// Retrieves the raw `VkSemaphore` of a timeline, to signal in your own
    /// `vkQueueSubmit` and to pass to
    /// [`exportGPUSemaphore()`](Self::exportGPUSemaphore).
    ///
    /// # Arguments
    ///
    /// * `timeline` - The timeline to query.
    ///
    /// Valid for as long as you hold a reference.
    pub getGPUTimelineSemaphore:
        unsafe extern "system-unwind" fn(timeline: *mut VSGPUTimeline) -> VkSemaphore,
    /// Waits on the host for a value of a timeline you signal yourself.
    ///
    /// # Arguments
    ///
    /// * `timeline` - The timeline to wait on.
    ///
    /// * `value` - The value to wait for.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// This is how a filter recording and submitting for itself waits;
    /// `vkWaitSemaphores` on the raw handle is not, for the reason
    /// [`gpuExecWaitValue()`](Self::gpuExecWaitValue) gives. See
    /// [`VSGPUDrainResult`].
    ///
    /// # Safety
    ///
    /// Waiting for a value an exec pool's timeline has not submitted is fatal,
    /// since nothing would ever signal it. A timeline you signal yourself has no
    /// bound the core could check, because another thread may be about to submit
    /// the value.
    ///
    /// This is not the way to wait out a whole exec pool:
    /// [`gpuExecPoolWaitIdle()`](Self::gpuExecPoolWaitIdle) does that and also
    /// promises the release callbacks have run.
    pub gpuTimelineWaitValue: unsafe extern "system-unwind" fn(
        timeline: *mut VSGPUTimeline,
        value: u64,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,

    // MARK: Memory from the core's pool

    /// Allocates scratch memory through the same sub allocator frame planes use,
    /// so it is counted against the VRAM limit, predicted by the thread pool's
    /// admission control and recycled through the size buckets.
    ///
    /// # Arguments
    ///
    /// * `core` - The core to allocate from.
    ///
    /// * `size` - The requested size in bytes.
    ///
    /// * `usage` - `VkBufferUsageFlags` for the buffer.
    ///
    /// * `requiredFlags` - `VkMemoryPropertyFlags` the memory must have.
    ///
    /// * `preferredFlags` - `VkMemoryPropertyFlags` the memory should have.
    ///
    /// * `info` - Filled in with the buffer and its mapping.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// Host visible requests come back persistently mapped. Returns `NULL` with
    /// the error set on failure.
    ///
    /// # Safety
    ///
    /// Unlike frames, buffers have no producer pair anyone waits on, so destroy
    /// only after the submissions using the buffer have completed on the device,
    /// and at the latest in the filter's free callback.
    pub createGPUBuffer: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        size: VkDeviceSize,
        usage: VkBufferUsageFlags,
        requiredFlags: VkMemoryPropertyFlags,
        preferredFlags: VkMemoryPropertyFlags,
        info: *mut VSVulkanBufferInfo,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> *mut VSGPUBuffer,
    /// Destroys a buffer created by
    /// [`createGPUBuffer()`](Self::createGPUBuffer).
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to destroy.
    pub destroyGPUBuffer: unsafe extern "system-unwind" fn(buffer: *mut VSGPUBuffer),
    /// Allocates a region of the pool frame planes come from, for a resource
    /// this API has no constructor for.
    ///
    /// # Arguments
    ///
    /// * `core` - The core to allocate from.
    ///
    /// * `requirements` - The `VkMemoryRequirements` of the resource.
    ///
    /// * `requiredFlags` - `VkMemoryPropertyFlags` the memory must have.
    ///
    /// * `preferredFlags` - `VkMemoryPropertyFlags` the memory should have.
    ///
    /// * `info` - Filled in with the memory, offset and mapping.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// Create the resource, ask Vulkan what memory it needs, allocate it here,
    /// and bind against the memory and offset that come back. Over a private
    /// `vkAllocateMemory` this buys everything the pool does: the region counts
    /// against the VRAM limit, is visible to admission control, is recycled
    /// through the size buckets and takes part in reclamation under pressure.
    ///
    /// # Safety
    ///
    /// Bind at [`offset`](VSVulkanMemoryInfo::offset), *not* at the start of
    /// [`memory`](VSVulkanMemoryInfo::memory). The region sits inside a shared
    /// block, and images are aligned further inside it since their requirements
    /// are routinely coarser than the pool's regions. The distance is reserved,
    /// so any alignment is satisfied at a bounded overshoot.
    ///
    /// Two resources must not bind here, and neither is detectable from the
    /// requirements, so both are the caller's to check: one created with
    /// external memory info, since these blocks are not exportable, and one
    /// whose `VkMemoryDedicatedRequirements` report
    /// `requiresDedicatedAllocation`, since a suballocated region is by
    /// definition not dedicated.
    ///
    /// A freed region returns to the pool at once and may back another resource
    /// immediately, so freeing early aliases live data rather than faulting.
    /// Free only once the submissions using the resource have completed, or hand
    /// it to [`gpuExecUsesMemory()`](Self::gpuExecUsesMemory).
    ///
    /// Returns `NULL` with the error set.
    pub allocateGPUMemory: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        requirements: *const VkMemoryRequirements,
        requiredFlags: VkMemoryPropertyFlags,
        preferredFlags: VkMemoryPropertyFlags,
        info: *mut VSVulkanMemoryInfo,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> *mut VSGPUMemory,
    /// Frees a region allocated by
    /// [`allocateGPUMemory()`](Self::allocateGPUMemory).
    ///
    /// # Arguments
    ///
    /// * `memory` - The region to free.
    pub freeGPUMemory: unsafe extern "system-unwind" fn(memory: *mut VSGPUMemory),
    /// Declares GPU memory the core did not allocate -- a CUDA pool, another
    /// Vulkan device, a video session -- into the same accounting, so the frame
    /// cache, admission control and the unified-memory brake all see it.
    ///
    /// # Arguments
    ///
    /// * `core` - The core the memory belongs to.
    ///
    /// * `bytes` - The absolute total currently held, not a delta.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// The core cannot veto memory it does not own, so a reservation is never
    /// refused; what an increase buys is cooperation. When the declared bytes
    /// push the pool past its limit, cached GPU frames are evicted to a tenth
    /// under it and idle allocator blocks handed back to the driver before the
    /// call returns, so reserve or update *before* the foreign allocation and
    /// the VRAM has actually been vacated. Shrinking and releasing just subtract.
    ///
    /// `bytes` is an absolute total, so publishing what you hold makes drift
    /// impossible however the calls interleave. Concurrent updates are safe and
    /// the last total wins; negative counts as zero.
    ///
    /// # Safety
    ///
    /// Only declare memory on the core's device -- match
    /// [`device_uuid`](VSVulkanCoreInfo::device_uuid) or
    /// [`device_luid`](VSVulkanCoreInfo::device_luid) from
    /// [`VSVulkanCoreInfo`] -- and never declare bytes the core already accounts
    /// ([`createGPUBuffer()`](Self::createGPUBuffer),
    /// [`allocateGPUMemory()`](Self::allocateGPUMemory), GPU frames) or they
    /// count twice. The exception is your imports of planes
    /// [`exportGPUPlane()`](Self::exportGPUPlane) handed out, which belong in
    /// the declaration where [`VS_GPU_IMPORT_CHARGED_AGAIN`] is 1.
    pub reserveGPUMemory: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        bytes: i64,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> *mut VSGPUMemoryReservation,
    /// Updates the byte total of a reservation.
    ///
    /// # Arguments
    ///
    /// * `reservation` - The reservation to update.
    ///
    /// * `bytes` - The new absolute total.
    ///
    /// Unlike [`releaseGPUMemoryReservation()`](Self::releaseGPUMemoryReservation)
    /// this keeps the handle, so it may overlap another call on the same
    /// reservation.
    pub updateGPUMemoryReservation:
        unsafe extern "system-unwind" fn(reservation: *mut VSGPUMemoryReservation, bytes: i64),
    /// Drops a reservation.
    ///
    /// # Arguments
    ///
    /// * `reservation` - The reservation to drop; `NULL` is ignored.
    ///
    /// # Safety
    ///
    /// Destroys the handle, so unlike an update it may not overlap another call
    /// on the same reservation. Belongs in the filter's free callback.
    pub releaseGPUMemoryReservation:
        unsafe extern "system-unwind" fn(reservation: *mut VSGPUMemoryReservation),

    // MARK: Shader compilation

    /// Compiles a shader at runtime, so plugins can ship readable kernel source
    /// instead of SPIR-V blobs.
    ///
    /// # Arguments
    ///
    /// * `core` - The core to compile against.
    ///
    /// * `language` - A [`VSGPUShaderLanguage`].
    ///
    /// * `source` - The shader source text.
    ///
    /// * `errorLog` - Buffer the compile log is written to on failure.
    ///
    /// * `errorLogSize` - Size of `errorLog`.
    ///
    /// Pure CPU work through the statically embedded glslang: no device is
    /// touched and no optimizer runs, so precompiled `-O` blobs remain the
    /// alternative for whoever wants them and both feed the same pipeline
    /// creation path. Results are cached per core by source text, so compiling
    /// the same kernel from many filter instances parses once.
    ///
    /// Specialize by concatenating a `#define` preamble in front of the kernel
    /// body; there is no include handler. Compute stage only. Returns `NULL` with
    /// the log filled on failure, including for languages this core does not
    /// know.
    pub compileGPUShader: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        language: c_int,
        source: *const c_char,
        errorLog: *mut c_char,
        errorLogSize: c_int,
    ) -> *mut VSGPUShader,
    /// Retrieves the SPIR-V words of a compiled shader.
    ///
    /// # Arguments
    ///
    /// * `shader` - The shader to query.
    ///
    /// * `sizeInBytes` - Filled in with the size of the returned words.
    pub getGPUShaderCode: unsafe extern "system-unwind" fn(
        shader: *const VSGPUShader,
        sizeInBytes: *mut usize,
    ) -> *const u32,
    /// Releases a compiled shader.
    ///
    /// # Arguments
    ///
    /// * `shader` - The shader to release.
    pub freeGPUShader: unsafe extern "system-unwind" fn(shader: *mut VSGPUShader),

    // MARK: Exec pools

    /// Creates an exec pool on one of the core's queues.
    ///
    /// # Arguments
    ///
    /// * `core` - The core to create the pool on.
    ///
    /// * `queue` - A [`VSVulkanQueueType`].
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// The core sizes the pool's context ring itself, from its worker thread
    /// count. Acquiring waits out the ring's oldest submission, and may
    /// additionally wait on the core's device-wide in-flight budget, which caps
    /// the total bytes submitted-but-unfinished work retains across all pools.
    ///
    /// # Safety
    ///
    /// A pool on [`Transfer`](VSVulkanQueueType::Transfer) may only ever record
    /// copies; a pool anything is dispatched into belongs on
    /// [`Compute`](VSVulkanQueueType::Compute).
    ///
    /// A pool belongs to *one* filter instance: create it in the filter's create
    /// function, destroy it in the free callback, and never share one between
    /// instances. Nothing about a pool needs sharing -- the ring is sized for the
    /// threads one instance is called on, and a dependency between two of them
    /// travels as a timeline value rather than as a shared recording.
    pub createGPUExecPool: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        queue: c_int,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> *mut VSGPUExecPool,
    /// Destroys an exec pool.
    ///
    /// # Arguments
    ///
    /// * `pool` - The pool to destroy.
    ///
    /// # Safety
    ///
    /// Drains the GPU first, so everything it still holds is released safely, and
    /// returns only once every release callback the pool registered has run, so
    /// instance data can go right after it. A context still held by any thread at
    /// that point is fatal.
    pub freeGPUExecPool: unsafe extern "system-unwind" fn(pool: *mut VSGPUExecPool),
    /// Blocks until every submission made through this pool has completed.
    ///
    /// # Arguments
    ///
    /// * `pool` - The pool to drain.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// Filters do not need this per frame -- producer pairs make consumers wait
    /// on the device instead -- but one shot setup work, such as uploading
    /// weights or tables a filter will read for the rest of its life, has to know
    /// the copy landed before recording anything that reads it. Also releases
    /// everything those submissions were keeping alive.
    ///
    /// Returns a [`VSGPUDrainResult`]. The error is set for both failures:
    /// [`DeviceLost`](VSGPUDrainResult::DeviceLost) is still a failure of the
    /// promise above, it is simply the failure after which nothing is running any
    /// more.
    ///
    /// # Safety
    ///
    /// Call it from a thread holding no context of the pool. Holding one is
    /// fatal, since that recording could never be covered and every worker doing
    /// the same would deadlock. It is a setup call, not a frame path one.
    pub gpuExecPoolWaitIdle: unsafe extern "system-unwind" fn(
        pool: *mut VSGPUExecPool,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Waits for one value on the pool's timeline instead of for the whole pool,
    /// so the frames running beside this one keep their submissions flowing.
    ///
    /// # Arguments
    ///
    /// * `pool` - The pool to wait on.
    ///
    /// * `value` - The value to wait for, as returned in `signaledValue` by
    ///   [`gpuExecSubmit()`](Self::gpuExecSubmit).
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// This is how a filter waits on the host. Do not wait on the timeline
    /// yourself with `vkWaitSemaphores`, because a bare wait cannot tell you the
    /// one thing that matters afterwards: a GPU reset releases every waiter by
    /// force-signalling the timelines past anything they could be waiting for, so
    /// a raw wait returns success immediately while the memory the dispatch was
    /// going to write still holds whatever it held before, and a frame built from
    /// it looks perfectly fine. This checks the value that actually satisfied the
    /// wait. See [`VSGPUDrainResult`].
    ///
    /// # Safety
    ///
    /// The value must be one this pool submitted, or anything below it. A value
    /// past everything the pool has ever queued is fatal rather than an unbounded
    /// wait on a signal that is never coming. A value the pool did allocate but
    /// whose submission failed is past checking and does hang.
    ///
    /// For a timeline of your own rather than a pool's, see
    /// [`gpuTimelineWaitValue()`](Self::gpuTimelineWaitValue).
    pub gpuExecWaitValue: unsafe extern "system-unwind" fn(
        pool: *mut VSGPUExecPool,
        value: u64,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Retrieves the pool's timeline as the counted object
    /// [`setGPUPlaneProducer()`](Self::setGPUPlaneProducer) takes, for publishing
    /// producer pairs by hand on frames the pool does not know about.
    ///
    /// # Arguments
    ///
    /// * `pool` - The pool to query.
    ///
    /// The out of order producer case is what this is for: the submission that
    /// wrote a plane was submitted calls ago and `signaledValue` was saved for
    /// this moment.
    ///
    /// The pool holds its own reference, so publishing it needs no reference of
    /// yours. [`getGPUTimelineSemaphore()`](Self::getGPUTimelineSemaphore) gives
    /// the raw handle when [`exportGPUSemaphore()`](Self::exportGPUSemaphore)
    /// needs one.
    ///
    /// # Safety
    ///
    /// *Never* signal that handle yourself: the pool allocates signal values
    /// under the queue lock at submit, and an external signal races them on a
    /// timeline where values must only increase. The pool compares the counter
    /// with the last value it submitted whenever it reaps, and a counter past it
    /// is fatal. Publish only values it has submitted.
    pub gpuExecPoolTimeline:
        unsafe extern "system-unwind" fn(pool: *mut VSGPUExecPool) -> *mut VSGPUTimeline,

    // MARK: Recording contexts

    /// Claims a context and begins recording.
    ///
    /// # Arguments
    ///
    /// * `pool` - The pool to record on.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// Returns `NULL` with the error set on device loss.
    ///
    /// # Safety
    ///
    /// Every acquire must end in exactly one submit or abandon, from the thread
    /// that acquired. Retaining, submitting or abandoning from any other thread
    /// is fatal, and so is any call with the handle after the submit or abandon
    /// that ended it.
    ///
    /// Do not keep a handle past the call that ended it. The handle lives in its
    /// ring slot for the life of the pool, so what the check tests is whether
    /// the slot's recording belongs to the calling thread. It does not once the
    /// recording ends -- until the same thread acquires the same slot again, at
    /// which point a kept handle is indistinguishable from the fresh one, being
    /// the same object, and the call lands on a recording it was never meant for.
    ///
    /// One context per thread, of this or any other pool: a ring has only two to
    /// eight contexts, so a second acquire from a thread already holding one
    /// could only wait for a slot that, once every worker did the same, nobody
    /// would ever release. It is fatal instead, reported when the second acquire
    /// would wait. A dependency between two pools is carried by a timeline value,
    /// so submit and release the first recording, then acquire the second and
    /// wait on the value the first signalled.
    pub gpuExecAcquire: unsafe extern "system-unwind" fn(
        pool: *mut VSGPUExecPool,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> *mut VSGPUExecContext,
    /// Retrieves the command buffer being recorded.
    ///
    /// # Arguments
    ///
    /// * `context` - The recording to query.
    ///
    /// Put anything Vulkan allows into it.
    pub gpuExecCommandBuffer:
        unsafe extern "system-unwind" fn(context: *mut VSGPUExecContext) -> VkCommandBuffer,
    /// Declares that this submission reads the frame.
    ///
    /// # Arguments
    ///
    /// * `context` - The recording to declare it on.
    ///
    /// * `frame` - The frame being read.
    ///
    /// Its planes' producer pairs become device side waits, and the frame is kept
    /// alive until the submission completes. Takes its own reference, so the
    /// caller still releases its own reference normally.
    ///
    /// # Safety
    ///
    /// Fatal on a frame with a plane handed to a foreign API; see
    /// [`VSVulkanExportedMemory`].
    pub gpuExecReadsFrame: unsafe extern "system-unwind" fn(
        context: *mut VSGPUExecContext,
        frame: *const VSFrame,
    ),
    /// Declares that this submission writes the plane.
    ///
    /// # Arguments
    ///
    /// * `context` - The recording to declare it on.
    ///
    /// * `frame` - The frame owning the plane.
    ///
    /// * `plane` - The plane index.
    ///
    /// [`gpuExecSubmit()`](Self::gpuExecSubmit) publishes the pool's
    /// `(timeline, value)` on it as the producer pair. Takes its own reference to
    /// the frame, as [`gpuExecReadsFrame()`](Self::gpuExecReadsFrame) does, so
    /// the caller still releases its own reference normally; it is dropped once
    /// the pair is published.
    ///
    /// # Safety
    ///
    /// The frame must own the plane outright. Copying a frame shares its planes,
    /// and a GPU plane has no copy on write, so a write declared on a shared
    /// plane would land in the other frame as well. That is fatal here rather
    /// than silent. Copy a frame only to inherit properties you then keep; to
    /// write, take [`newGPUVideoFrame()`](Self::newGPUVideoFrame) or
    /// [`newVideoFrame2()`](crate::VSAPI::newVideoFrame2) with the old frame as
    /// the property source. Sharing a plane this filter does not write stays
    /// fine, the check being per plane.
    ///
    /// Fatal on a plane handed to a foreign API, like
    /// [`gpuExecReadsFrame()`](Self::gpuExecReadsFrame).
    pub gpuExecWritesPlane: unsafe extern "system-unwind" fn(
        context: *mut VSGPUExecContext,
        frame: *mut VSFrame,
        plane: c_int,
    ),
    /// Hands a scratch buffer to the context, which destroys it once the
    /// submission completes.
    ///
    /// # Arguments
    ///
    /// * `context` - The recording to hand it to.
    ///
    /// * `buffer` - The buffer to hand over.
    ///
    /// # Safety
    ///
    /// Ownership transfers; do not destroy it yourself.
    pub gpuExecUsesBuffer: unsafe extern "system-unwind" fn(
        context: *mut VSGPUExecContext,
        buffer: *mut VSGPUBuffer,
    ),
    /// [`gpuExecUsesBuffer()`](Self::gpuExecUsesBuffer) for a bare region.
    ///
    /// # Arguments
    ///
    /// * `context` - The recording to hand it to.
    ///
    /// * `memory` - The region to hand over.
    ///
    /// The region stays out of the pool until the submission being recorded has
    /// completed, and its bytes count against the in-flight retention budget
    /// meanwhile. Release follows the ordinary retention timing described at
    /// [`createGPUExecPool()`](Self::createGPUExecPool), so what is guaranteed
    /// is that the region is never recycled early, not that it returns at a
    /// particular moment.
    ///
    /// # Safety
    ///
    /// Ownership transfers; do not free it yourself.
    ///
    /// The resource bound to the region is still yours to destroy, since this
    /// never saw it. A resource whose lifetime is the filter's -- created once,
    /// reused every frame -- is the easy case and wants no retention at all. One
    /// created per frame has to outlive its submission too, so keep those handles
    /// and destroy them when the filter is freed rather than trying to guess when
    /// the submission retired.
    pub gpuExecUsesMemory: unsafe extern "system-unwind" fn(
        context: *mut VSGPUExecContext,
        memory: *mut VSGPUMemory,
    ),
    /// Registers cleanup of your own on the retention list, for what the typed
    /// calls above cannot name.
    ///
    /// # Arguments
    ///
    /// * `context` - The recording to register it on.
    ///
    /// * `release` - Called once the submission completes. `NULL` is allowed.
    ///
    /// * `object` - Passed to `release`.
    ///
    /// * `bytes` - What `object` pins in device memory, counted against the
    ///   in-flight retention budget from submit until release. Pass 0 for host
    ///   side bookkeeping, and do not double count bytes a typed call already
    ///   counted for the same object.
    ///
    /// A per frame image is the case: the image, every view recorded against it
    /// and the region underneath must retire together, and one registration of a
    /// struct holding all three frees them in the right order -- views, image,
    /// then [`freeGPUMemory()`](Self::freeGPUMemory) -- where a call per object
    /// could not express the ordering.
    ///
    /// # Safety
    ///
    /// Call it between acquire and submit; abandon runs it immediately.
    ///
    /// The callback runs on whichever thread reaps the submission, so it must be
    /// thread safe. No core lock is held, so it may take its own locks and free
    /// whatever the object owns, and that is all it may do: acquiring a context,
    /// allocating GPU memory, and creating, freeing or waiting on a pool from
    /// inside a release callback are fatal, stopping the process with a message
    /// naming the call. The reaping thread may be one inside your own `getFrame`
    /// (a gated acquire or a failed allocation sweeps), so do not hold a lock
    /// across those calls that the callback also takes.
    pub gpuExecRetain: unsafe extern "system-unwind" fn(
        context: *mut VSGPUExecContext,
        release: VSGPUReleaseFunc,
        object: *mut c_void,
        bytes: VkDeviceSize,
    ),
    /// Ends recording and submits, then publishes the producer pairs.
    ///
    /// # Arguments
    ///
    /// * `context` - The recording to submit.
    ///
    /// * `signaledValue` - When non-`NULL`, receives the value this submission
    ///   signals on the pool's timeline.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// The timeline value is allocated inside the queue lock, so signals reach
    /// the queue in increasing order.
    ///
    /// Pass `signaledValue` to [`gpuExecWaitValue()`](Self::gpuExecWaitValue) to
    /// wait for exactly this submission, which is what a filter reading results
    /// back on the host wants:
    /// [`gpuExecPoolWaitIdle()`](Self::gpuExecPoolWaitIdle) also works but waits
    /// the pool's newest submission, so concurrent frames serialize on each
    /// other's work. Filters that only produce planes never need any of it; the
    /// producer pairs carry the synchronization.
    ///
    /// # Safety
    ///
    /// The context is consumed either way, and the handle is dead from then on:
    /// using it again is fatal. Returns nonzero with the error set on failure.
    pub gpuExecSubmit: unsafe extern "system-unwind" fn(
        context: *mut VSGPUExecContext,
        signaledValue: *mut u64,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Gives up a recording without submitting.
    ///
    /// # Arguments
    ///
    /// * `context` - The recording to abandon.
    ///
    /// Everything retained is released at once.
    pub gpuExecAbandon: unsafe extern "system-unwind" fn(context: *mut VSGPUExecContext),

    // MARK: Sharing frames with other APIs

    /// Exports the allocation backing a GPU frame plane as an opaque handle.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame owning the plane.
    ///
    /// * `plane` - The plane index.
    ///
    /// * `out` - Filled in with the exported handle and its identity.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// See [`VSVulkanExportedMemory`] for the identity, ownership, hand-off and
    /// synchronization rules.
    ///
    /// # Safety
    ///
    /// Only available when
    /// [`export_handle_type`](VSVulkanCoreInfo::export_handle_type) is nonzero.
    /// Fails on CPU frames, missing planes and devices without export support.
    pub exportGPUPlane: unsafe extern "system-unwind" fn(
        frame: *const VSFrame,
        plane: c_int,
        out: *mut VSVulkanExportedMemory,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Exports a timeline semaphore as an opaque handle.
    ///
    /// # Arguments
    ///
    /// * `core` - The core the semaphore belongs to.
    ///
    /// * `semaphore` - The `VkSemaphore` to export.
    ///
    /// * `out` - Filled in with the exported handle.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// See [`VSVulkanExportedSemaphore`].
    ///
    /// Use it on the `ready_semaphore` of a plane taken with
    /// [`getExportableFrameFilter()`](Self::getExportableFrameFilter), always one
    /// of the core's own timelines, all of which are created exportable when the
    /// capability exists, or on your own timeline created with
    /// `VkExportSemaphoreCreateInfo`, to signal your producer pairs from the
    /// foreign API.
    ///
    /// # Safety
    ///
    /// On Windows, the first call for a timeline the core created -- its own or
    /// one from [`createGPUTimeline()`](Self::createGPUTimeline) -- exports it,
    /// and every call returns a duplicate of that one handle. A semaphore you
    /// created yourself is exported anew on every call, which Vulkan allows only
    /// once per semaphore for an NT handle: export it once and duplicate that
    /// handle if you need more. Only available when
    /// [`semaphore_export_handle_type`](VSVulkanCoreInfo::semaphore_export_handle_type)
    /// is nonzero.
    pub exportGPUSemaphore: unsafe extern "system-unwind" fn(
        core: *mut VSCore,
        semaphore: VkSemaphore,
        out: *mut VSVulkanExportedSemaphore,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Host waits every plane's producer pair *and* makes the completed writes
    /// available outside the device's own domain, which is the part a bare
    /// producer wait does not give you.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame to make available.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// The spec only defines cross device visibility through external semaphores
    /// or an availability chain like this one, so skipping it means relying on
    /// driver behavior. Call once per frame before reading it through an exported
    /// handle; the cost is one submission round trip. Frames only read through
    /// Vulkan on the same device never need this -- the producer pairs carry the
    /// dependency there.
    pub waitGPUFrame: unsafe extern "system-unwind" fn(
        frame: *const VSFrame,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> c_int,
    /// Takes a requested input frame for a foreign API, the way
    /// [`getFrameFilter()`](crate::VSAPI::getFrameFilter) takes one for the
    /// core's queues.
    ///
    /// # Arguments
    ///
    /// * `n` - The frame number being requested.
    ///
    /// * `node` - The node to request from.
    ///
    /// * `frameCtx` - The frame context of the calling filter.
    ///
    /// * `errorMessage` - Buffer the error is written to on failure.
    ///
    /// * `errorMessageSize` - Size of `errorMessage`.
    ///
    /// The frame comes back with every plane handed to the foreign side with its
    /// contents and ready for [`exportGPUPlane()`](Self::exportGPUPlane), its
    /// producer pairs naming the hand-over. Wait on those, or call
    /// [`waitGPUFrame()`](Self::waitGPUFrame), before the foreign side touches
    /// it. Where nothing else holds the frame -- this filter its only consumer,
    /// requesting `rpStrictSpatial` or `rpNoFrameReuse`, so the source keeps no
    /// cache -- the planes are handed over in place; otherwise the frame is
    /// copied first, one GPU copy per plane, on the compute queue. That copy is
    /// made once per frame and shared by every request of it, from any filter,
    /// for as long as the frame stays cached, so a temporal filter pays one copy
    /// per source frame rather than one per tap.
    ///
    /// # Safety
    ///
    /// Like any input frame it is read only, on the foreign side as much as on
    /// the core's queues, and it never goes back to the core: returning it, or a
    /// frame that shares its planes or holds it in its properties, fails that
    /// frame with an error, and
    /// [`cacheFrame()`](crate::VSAPI::cacheFrame) leaves such a frame out of the
    /// cache with a warning. Return a new frame, and take a frame you pass
    /// through with [`getFrameFilter()`](crate::VSAPI::getFrameFilter) instead,
    /// freeing that before calling this if you process it after all.
    ///
    /// Keep it until the foreign reads are done, or publish their completion with
    /// [`setGPUPlaneProducer()`](Self::setGPUPlaneProducer), which the core waits
    /// for before the memory is reused. Declaring it in an exec context is fatal,
    /// as for any plane handed over.
    ///
    /// Frames in its properties are not prepared: take an `_Alpha` frame as a
    /// clip of its own (`std.PropToClip`) and request that.
    ///
    /// Consumes one request of the frame for this activation:
    /// [`getFrameFilter()`](crate::VSAPI::getFrameFilter) and this return `NULL`
    /// for it afterwards unless it was requested again, as a temporal filter
    /// clamping at the clip's ends requests one frame twice, each request giving
    /// a frame. Call [`getFrameFilter()`](crate::VSAPI::getFrameFilter) first to
    /// keep the original as well, which makes this a copy.
    ///
    /// Fails on CPU frames and on devices without export support, and returns
    /// `NULL` for a frame that was not requested. Fatal when the calling thread
    /// holds an exec context, since preparing takes one of the core's.
    pub getExportableFrameFilter: unsafe extern "system-unwind" fn(
        n: c_int,
        node: *mut VSNode,
        frameCtx: *mut VSFrameContext,
        errorMessage: *mut c_char,
        errorMessageSize: c_int,
    ) -> *const VSFrame,
}

#[cfg(test)]
mod tests {
    use core::mem::{offset_of, size_of};

    use super::*;

    /// [`VSVULKANAPI`] is a plain run of function pointers, exactly like
    /// [`VSAPI`](crate::VSAPI), and the library always exposes the full struct.
    /// A client is therefore only ever allowed to declare a *prefix* of it, so a
    /// miscounted or mis-gated member silently shifts everything after it.
    ///
    /// The count is `struct VSVULKANAPI`'s member count in `VSVulkan4.h`.
    #[test]
    fn vsvulkanapi_member_count() {
        const MEMBERS: usize = 44;
        assert_eq!(
            size_of::<VSVULKANAPI>(),
            MEMBERS * size_of::<*const ()>(),
            "VSVULKANAPI should have {MEMBERS} members"
        );
    }

    /// Sizes and offsets exactly as the C compiler reports them for
    /// `VSVulkan4.h`. Regenerate the numbers with a `sizeof`/`offsetof` probe
    /// over the same headers if one of these ever needs to change -- a change
    /// here is a change to the ABI the core hands out, not a refactor.
    #[test]
    fn layout() {
        assert_eq!(size_of::<VSVulkanCoreHandles>(), 48);
        assert_eq!(offset_of!(VSVulkanCoreHandles, get_instance_proc_addr), 24);
        assert_eq!(offset_of!(VSVulkanCoreHandles, compute_queue_family), 32);
        assert_eq!(offset_of!(VSVulkanCoreHandles, transfer_queue_index), 44);

        assert_eq!(size_of::<VSVulkanPlaneInfo>(), 32);
        assert_eq!(offset_of!(VSVulkanPlaneInfo, buffer_size), 8);
        assert_eq!(offset_of!(VSVulkanPlaneInfo, ready_semaphore), 16);
        assert_eq!(offset_of!(VSVulkanPlaneInfo, ready_value), 24);

        assert_eq!(size_of::<VSVulkanBufferInfo>(), 40);
        assert_eq!(offset_of!(VSVulkanBufferInfo, address), 8);
        assert_eq!(offset_of!(VSVulkanBufferInfo, mapped), 16);
        assert_eq!(offset_of!(VSVulkanBufferInfo, memory_flags), 32);

        assert_eq!(size_of::<VSVulkanMemoryInfo>(), 40);
        assert_eq!(offset_of!(VSVulkanMemoryInfo, memory), 0);
        assert_eq!(offset_of!(VSVulkanMemoryInfo, mapped), 24);

        assert_eq!(size_of::<VSVulkanExportedMemory>(), 56);
        assert_eq!(offset_of!(VSVulkanExportedMemory, memory_type_index), 16);
        assert_eq!(offset_of!(VSVulkanExportedMemory, handle), 48);

        assert_eq!(size_of::<VSVulkanExportedSemaphore>(), 16);
        assert_eq!(offset_of!(VSVulkanExportedSemaphore, handle), 8);

        assert_eq!(size_of::<VSVulkanCoreInfo>(), 336);
        assert_eq!(offset_of!(VSVulkanCoreInfo, device_uuid), 288);
        assert_eq!(offset_of!(VSVulkanCoreInfo, device_luid), 304);
        assert_eq!(offset_of!(VSVulkanCoreInfo, export_handle_type), 320);
        assert_eq!(offset_of!(VSVulkanCoreInfo, unified_memory), 328);

        assert_eq!(size_of::<VSVulkanDeviceListEntry>(), 568);
        assert_eq!(offset_of!(VSVulkanDeviceListEntry, unusable_reason), 276);
        assert_eq!(offset_of!(VSVulkanDeviceListEntry, device_luid_valid), 560);

        // The enumerations cross the ABI as C `int`s.
        assert_eq!(size_of::<VSVulkanQueueType>(), size_of::<c_int>());
        assert_eq!(size_of::<VSGPUDrainResult>(), size_of::<c_int>());
        assert_eq!(size_of::<VSGPUShaderLanguage>(), size_of::<c_int>());
        assert_eq!(size_of::<VSVulkanLevel>(), size_of::<c_int>());
        assert_eq!(size_of::<VSVulkanRequirement>(), size_of::<c_int>());
    }

    /// The header writes the safe-to-destroy question as the positive condition
    /// on purpose, so a value added later is unsafe by default.
    #[test]
    fn drain_result_safety() {
        assert!(gpu_drain_safe_to_destroy(VSGPUDrainResult::Drained));
        assert!(gpu_drain_safe_to_destroy(VSGPUDrainResult::DeviceLost));
        assert!(!gpu_drain_safe_to_destroy(VSGPUDrainResult::Incomplete));
    }

    /// Compiling against the header only proves the layout is self-consistent.
    /// `enumerateVulkanDevices` takes no core, so it is the cheapest call that
    /// would fault loudly if a member had moved relative to the library.
    #[cfg(feature = "link-vs")]
    #[test]
    fn vulkan_api_is_reachable() {
        let api = unsafe { crate::getVapourSynthAPI(crate::VAPOURSYNTH_API_VERSION) };
        assert!(
            !api.is_null(),
            "linked library does not support this API version"
        );

        let vk = unsafe { ((*api).getVulkanAPI)() };
        assert!(!vk.is_null(), "getVulkanAPI must never return null");

        let mut error = [0 as c_char; 512];
        let count = unsafe {
            ((*vk).enumerateVulkanDevices)(
                core::ptr::null_mut(),
                0,
                error.as_mut_ptr(),
                error.len() as c_int,
            )
        };
        assert!(
            count >= 0,
            "enumerateVulkanDevices failed: {}",
            unsafe { core::ffi::CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
        );
    }
}
