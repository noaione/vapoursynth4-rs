/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

//! Safe wrappers for the GPU side of the API, [`VSVULKANAPI`](ffi::VSVULKANAPI).
//!
//! Everything here is behind the `vs-vulkan` feature, which implies `vs-43`,
//! because the whole GPU API is one of the API 4.3 additions. The API table
//! itself is reachable from any core, so there is no version to negotiate and no
//! null to check: [`Api::vulkan()`] always returns a usable table.
//!
//! The module splits along the line between what the library checks and what it
//! cannot:
//!
//! * **Owned resources, device information and frames.** A [`Timeline`], a
//!   [`Shader`], a [`GpuBuffer`], a [`GpuMemory`] region, a
//!   [`GpuMemoryReservation`] and an [`ExecPool`] each hold one handle and give
//!   it back when dropped. Device and core information comes back as plain owned
//!   data instead of the fixed size C structs, and GPU frames are created with
//!   [`Core::new_gpu_video_frame()`].
//!
//! * **Everything that records or publishes GPU work.** Acquiring a recording,
//!   declaring what it reads, writes and retains, submitting it, locking a queue,
//!   publishing a plane's producer pair and taking an input frame with
//!   [`VideoNode::get_exportable_frame_filter()`] are `unsafe`: most are fatal
//!   when misused rather than merely wrong, and none of them can be checked
//!   against the device state the wrapper would need to see.
//!
//! # Lifetime rules the wrapper cannot enforce
//!
//! Handing a buffer or a memory region to a submission transfers ownership, and
//! the call that does it is `unsafe`; so is submitting. Dropping a [`GpuBuffer`]
//! or a [`GpuMemory`] that a submission is still using aliases live data rather
//! than faulting, exactly as the C API documents, so the promise that the
//! submission has completed has to be made at the `unsafe` call site that used
//! it. Frames are the exception: their producer pairs carry the synchronization,
//! which is why creating and reading them is safe.

use std::{
    cell::Cell,
    ffi::{CStr, c_char, c_void},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{MaybeUninit, forget, size_of_val},
    ops::Deref,
    ptr::{NonNull, null, null_mut},
    slice,
};

use thiserror::Error;

use crate::{
    api::Api,
    core::Core,
    ffi,
    frame::{Frame, FrameContext, VideoFormat, VideoFrame, internal::FrameFromPtr},
    node::{Node, VideoNode},
};

/// The size of the error buffers the API writes its messages into.
///
/// The API takes the size alongside the pointer and truncates rather than
/// overflowing, so this only decides how much of a message survives.
///
/// The array type below spells the same number, since an array extent cannot
/// be an `i32` without a cast.
const ERROR_BUFFER_SIZE: i32 = 512;

/// A buffer the API writes an error message into.
type ErrorBuffer = [c_char; 512];

/// Calls `f` with a fresh error buffer, and reads the message back out.
///
/// Every failure in this module is reported the same way, so every call site
/// goes through here. The message is copied out immediately because the library
/// only guarantees it until the next call.
fn with_error_buffer<T>(f: impl FnOnce(*mut c_char, i32) -> T) -> (T, Option<String>) {
    let mut buffer: ErrorBuffer = [0; 512];
    let result = f(buffer.as_mut_ptr(), ERROR_BUFFER_SIZE);
    (result, read_message(&buffer))
}

/// Copies a NUL terminated message out of an error buffer.
fn read_message(buffer: &ErrorBuffer) -> Option<String> {
    let bytes = buffer
        .iter()
        .take_while(|byte| **byte != 0)
        .map(|byte| byte.cast_unsigned())
        .collect::<Vec<_>>();

    (!bytes.is_empty()).then(|| String::from_utf8_lossy(&bytes).into_owned())
}

/// Reads a NUL terminated C string out of a fixed size character array.
fn read_string<const N: usize>(buffer: &[c_char; N]) -> String {
    let bytes = buffer
        .iter()
        .take_while(|byte| **byte != 0)
        .map(|byte| byte.cast_unsigned())
        .collect::<Vec<_>>();

    String::from_utf8_lossy(&bytes).into_owned()
}

/// Converts the `c_int` a drain call reports into a [`ffi::VSGPUDrainResult`].
///
/// The C API returns a plain `int` for these, so a value outside the documented
/// range is a library bug rather than something a caller can act on.
fn drain_result(value: i32) -> Option<ffi::VSGPUDrainResult> {
    match value {
        0 => Some(ffi::VSGPUDrainResult::Drained),
        1 => Some(ffi::VSGPUDrainResult::Incomplete),
        2 => Some(ffi::VSGPUDrainResult::DeviceLost),
        _ => None,
    }
}

/// An error reported by the GPU side of the API.
///
/// Most GPU calls do not return status codes; they either return a handle or
/// write a human readable message out. This carries that message, and falls back
/// to a description of the call when the library left the buffer empty.
#[derive(Error, Debug, Clone, PartialEq, Eq, Hash)]
#[error("{message}")]
pub struct VulkanError {
    message: String,
}

impl VulkanError {
    /// Wraps a message the library wrote, or `fallback` when it wrote nothing.
    fn new(message: Option<String>, fallback: &str) -> Self {
        Self {
            message: message.unwrap_or_else(|| fallback.to_owned()),
        }
    }

    /// The message the library wrote, or the fallback it was constructed with.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// A [`VSVULKANAPI`](ffi::VSVULKANAPI) table.
///
/// The table is looked up once per call through
/// [`Api::getVulkanAPI`](ffi::VSAPI::getVulkanAPI), which is a field read and
/// never fails; there is no version negotiation on this side of the API, and the
/// table grows with the core API instead of being capped by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VsVulkanApi(*const ffi::VSVULKANAPI);

impl VsVulkanApi {
    /// Fetches the GPU API table of a linked library.
    ///
    /// [`Api::vulkan()`] is the same call with less typing.
    #[must_use]
    pub fn new(api: &Api) -> Self {
        // SAFETY: the API table is valid as long as the library is, and this
        // member is one of the API 4.3 additions the feature guarantees.
        Self(unsafe { (api.getVulkanAPI)() })
    }

    /// Enumerates every physical device the Vulkan loader can see.
    ///
    /// This is the device list [`Core::set_vulkan_device()`] indexes into, and
    /// it is available before a core exists, which is the point: a filter that
    /// wants a particular GPU looks first and creates its core afterwards.
    ///
    /// `usable` entries are the ones that expose everything the API needs; the
    /// unusable ones carry the reason in
    /// [`VulkanDeviceInfo::unusable_reason`].
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the loader could not be reached at all.
    pub fn enumerate_devices(&self) -> Result<Vec<VulkanDeviceInfo>, VulkanError> {
        let (count, message) = with_error_buffer(|buffer, size| unsafe {
            (self.enumerateVulkanDevices)(null_mut(), 0, buffer, size)
        });

        if count < 0 {
            return Err(VulkanError::new(
                message,
                "could not enumerate Vulkan devices",
            ));
        }

        // A negative count was rejected above, and a count that does not fit in
        // a `usize` could not have come from a device list.
        let capacity = usize::try_from(count).unwrap_or(0);
        let mut entries =
            vec![unsafe { std::mem::zeroed::<ffi::VSVulkanDeviceListEntry>() }; capacity];
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let (written, message) = with_error_buffer(|buffer, size| unsafe {
            (self.enumerateVulkanDevices)(entries.as_mut_ptr(), count, buffer, size)
        });

        if written < 0 {
            return Err(VulkanError::new(
                message,
                "could not enumerate Vulkan devices",
            ));
        }

        // A negative count was rejected above.
        let written = usize::try_from(written).unwrap_or(0);
        entries.truncate(written.min(capacity));

        Ok(entries.iter().map(VulkanDeviceInfo::from).collect())
    }

    /// The raw API table.
    #[must_use]
    pub fn as_ptr(&self) -> *const ffi::VSVULKANAPI {
        self.0
    }
}

impl Deref for VsVulkanApi {
    type Target = ffi::VSVULKANAPI;

    fn deref(&self) -> &Self::Target {
        // SAFETY: the table outlives the library, and `Api` hands out this same
        // pointer for as long as the library is loaded.
        unsafe { &*self.0 }
    }
}

impl Api {
    /// The GPU API table of this library.
    ///
    /// The table is a compile time constant of the linked library, so there is
    /// nothing to negotiate and no option to handle.
    #[must_use]
    pub fn vulkan(&self) -> VsVulkanApi {
        VsVulkanApi::new(self)
    }
}

/// One entry of [`VsVulkanApi::enumerate_devices()`].
///
/// The C struct is a fixed set of 256 byte character arrays; this is the same
/// information with owned strings, so it can outlive the call and be stored.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VulkanDeviceInfo {
    /// The device name as the driver reports it.
    pub name: String,
    /// The highest Vulkan version the device supports, as a `VK_MAKE_VERSION`
    /// style integer.
    pub api_version: u32,
    /// A `VkPhysicalDeviceType`: integrated, discrete, virtual or CPU.
    pub device_type: i32,
    /// The device's local memory in bytes, or a negative estimate when the
    /// driver does not report one that can be trusted.
    pub device_memory: i64,
    /// Whether the device exposes everything the API needs.
    pub usable: bool,
    /// Why the device is not usable, when it is not. `None` for a usable device.
    pub unusable_reason: Option<String>,
    /// The device UUID, as [`ffi::VK_UUID_SIZE`] bytes.
    pub uuid: [u8; ffi::VK_UUID_SIZE],
    /// The device LUID, as [`ffi::VK_LUID_SIZE`] bytes.
    pub luid: [u8; ffi::VK_LUID_SIZE],
    /// The bitmask of NUMA nodes this device is local to.
    pub node_mask: u32,
    /// Whether [`Self::luid`] is meaningful.
    pub luid_valid: bool,
}

impl From<&ffi::VSVulkanDeviceListEntry> for VulkanDeviceInfo {
    fn from(entry: &ffi::VSVulkanDeviceListEntry) -> Self {
        let reason = read_string(&entry.unusable_reason);

        Self {
            name: read_string(&entry.device_name),
            api_version: entry.api_version,
            device_type: entry.device_type,
            device_memory: entry.device_memory,
            usable: entry.usable != 0,
            unusable_reason: (!reason.is_empty()).then_some(reason),
            uuid: entry.device_uuid,
            luid: entry.device_luid,
            node_mask: entry.device_node_mask,
            luid_valid: entry.device_luid_valid != 0,
        }
    }
}

/// What [`Core::vulkan_core_info()`] reports about the device a core is on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VulkanCoreInfo {
    /// The device name as the driver reports it.
    pub device_name: String,
    /// The device's total local memory in bytes.
    pub device_memory: i64,
    /// The driver's own budget for this process, if it reports one.
    pub budget: i64,
    /// How much of the budget is already allocated.
    pub allocated: i64,
    /// The limit the core itself enforces, as set by
    /// [`Core::set_max_vram_use()`].
    pub limit: i64,
    /// The device UUID, as [`ffi::VK_UUID_SIZE`] bytes.
    pub device_uuid: [u8; ffi::VK_UUID_SIZE],
    /// The device LUID, as [`ffi::VK_LUID_SIZE`] bytes.
    pub device_luid: [u8; ffi::VK_LUID_SIZE],
    /// The bitmask of NUMA nodes this device is local to.
    pub device_node_mask: u32,
    /// Whether [`Self::device_luid`] is meaningful.
    pub device_luid_valid: bool,
    /// The `VkExternalMemoryHandleTypeFlagBits` frame planes are exported with,
    /// or zero when the device cannot export them.
    pub export_handle_type: i32,
    /// The `VkExternalSemaphoreHandleTypeFlagBits` timelines are exported with,
    /// or zero when the device cannot export them.
    pub semaphore_export_handle_type: i32,
    /// Whether device memory is host memory.
    pub unified_memory: bool,
}

impl From<&ffi::VSVulkanCoreInfo> for VulkanCoreInfo {
    fn from(info: &ffi::VSVulkanCoreInfo) -> Self {
        Self {
            device_name: read_string(&info.device_name),
            device_memory: info.device_memory,
            budget: info.budget,
            allocated: info.allocated,
            limit: info.limit,
            device_uuid: info.device_uuid,
            device_luid: info.device_luid,
            device_node_mask: info.device_node_mask,
            device_luid_valid: info.device_luid_valid != 0,
            export_handle_type: info.export_handle_type,
            semaphore_export_handle_type: info.semaphore_export_handle_type,
            unified_memory: info.unified_memory != 0,
        }
    }
}

/// A borrowed handle to a [`Timeline`], owning no reference of its own.
///
/// [`ExecPool::timeline()`] hands one out for the pool's own timeline, which the
/// pool keeps alive; a [`Timeline`] you created yourself converts with
/// [`Timeline::as_timeline_ref()`]. Publishing a value on it does not extend its
/// lifetime, so whichever of the two it came from must outlive every frame it
/// was published on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimelineRef(*mut ffi::VSGPUTimeline);

impl TimelineRef {
    /// The raw handle.
    #[must_use]
    pub fn as_ptr(self) -> *mut ffi::VSGPUTimeline {
        self.0
    }
}

/// A timeline semaphore, used to order GPU work across queues and APIs.
///
/// A timeline you create starts at zero and is exportable wherever the device
/// allows it. It is published on frame planes with
/// [`VideoFrame::set_gpu_plane_producer()`] and waited on with
/// [`Timeline::wait_value()`]; the wait pins the semaphore as a `f64`.
///
/// The handle is reference counted. [`Timeline::clone()`] takes another
/// reference, and the last owner releases it, so a timeline can be handed to
/// something with its own lifetime without being tied to the first owner's.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Timeline {
    handle: NonNull<ffi::VSGPUTimeline>,
    api: VsVulkanApi,
}

// SAFETY: the timeline is a reference counted object owned by the core, not
// tied to the thread that created it. Publishing it on a frame and releasing it
// from whichever thread drops it are both ordinary refcount traffic.
unsafe impl Send for Timeline {}

impl Timeline {
    /// Wraps a freshly created handle that this value takes ownership of.
    fn from_owned(handle: *mut ffi::VSGPUTimeline, api: VsVulkanApi) -> Option<Self> {
        NonNull::new(handle).map(|handle| Self { handle, api })
    }

    /// The raw semaphore, for signalling in your own `vkQueueSubmit` and for
    /// handing to [`Core::export_gpu_semaphore()`].
    ///
    /// Valid for as long as you hold a reference.
    #[must_use]
    pub fn semaphore(&self) -> ffi::VkSemaphore {
        // SAFETY: the handle is live for as long as `self`.
        unsafe { (self.api.getGPUTimelineSemaphore)(self.handle.as_ptr()) }
    }

    /// A borrowed view of this timeline, for publishing on a frame plane.
    #[must_use]
    pub fn as_timeline_ref(&self) -> TimelineRef {
        TimelineRef(self.handle.as_ptr())
    }

    /// The raw handle.
    #[must_use]
    pub fn as_ptr(&self) -> *mut ffi::VSGPUTimeline {
        self.handle.as_ptr()
    }

    /// Waits on the host for one value of this timeline.
    ///
    /// This is how a filter that records and submits for itself waits. Waiting
    /// with `vkWaitSemaphores` on the raw semaphore is not equivalent: a GPU
    /// reset force-signals every timeline past anything a waiter could be
    /// waiting for, so a bare wait returns success at once even though the
    /// memory the dispatch was going to write still holds what it held before.
    /// This reports what value actually satisfied the wait, and retries an
    /// allocation failure inside the wait rather than counting it as completion.
    ///
    /// A GPU reset makes this return [`ffi::VSGPUDrainResult::DeviceLost`],
    /// which is the one result that makes everything the submission named safe
    /// to destroy; [`ffi::VSGPUDrainResult::Incomplete`] leaves it queued and
    /// untouchable.
    ///
    /// # Safety
    ///
    /// The value must be one that this timeline will actually reach. Waiting for
    /// a value an exec pool's timeline has not submitted is fatal, because
    /// nothing will ever signal it.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the wait could not be established at all, or if
    /// the library reported a drain result outside the documented set.
    pub unsafe fn wait_value(&self, value: u64) -> Result<ffi::VSGPUDrainResult, VulkanError> {
        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (self.api.gpuTimelineWaitValue)(self.handle.as_ptr(), value, buffer, size)
        });

        drain_result(result).ok_or_else(|| {
            VulkanError::new(
                message,
                "waiting on a timeline reported an unknown drain result",
            )
        })
    }
}

impl Clone for Timeline {
    fn clone(&self) -> Self {
        // SAFETY: the handle is live, and the added reference is released by the
        // `Drop` of the clone.
        unsafe { (self.api.addGPUTimelineRef)(self.handle.as_ptr()) };

        Self {
            handle: self.handle,
            api: self.api,
        }
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: every `Timeline` owns exactly one reference, and the last one
        // releases the object. Planes it was published on hold their own.
        unsafe { (self.api.freeGPUTimeline)(self.handle.as_ptr()) };
    }
}

/// A shader compiled at runtime, holding the SPIR-V words.
///
/// Compilation goes through the statically embedded glslang, so no device is
/// touched and no optimizer runs. Results are cached per core by source text, so
/// compiling the same kernel from many filter instances parses it once and every
/// handle shares the cached words.
///
/// The handle is independent of everything else once returned: it stays valid
/// after the core that compiled it is freed.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Shader {
    handle: NonNull<ffi::VSGPUShader>,
    api: VsVulkanApi,
}

// SAFETY: the shader holds the compiled words and nothing device specific, and
// is reference counted inside the core's cache.
unsafe impl Send for Shader {}
// SAFETY: a compiled shader is immutable once returned -- the only thing a
// `&Shader` reaches is [`Shader::code()`], which reads the cached words -- so
// sharing one with the worker threads of a filter is safe. A parallel GPU
// filter holds one in its instance for exactly that.
unsafe impl Sync for Shader {}

impl Shader {
    /// The SPIR-V words.
    ///
    /// The slice is valid for as long as this shader is, and is empty only if
    /// the library reported no code, which it does not do for a live handle.
    #[must_use]
    pub fn code(&self) -> &[u32] {
        let mut size = 0usize;
        // SAFETY: the handle is live for as long as `self`, and the returned
        // words are owned by the shader.
        let ptr = unsafe { (self.api.getGPUShaderCode)(self.handle.as_ptr(), &raw mut size) };

        if ptr.is_null() || size < size_of::<u32>() {
            return &[];
        }

        // SAFETY: the library reports a byte count for a `uint32_t` array, so
        // the pointer is aligned and covers `size` readable bytes.
        unsafe { slice::from_raw_parts(ptr, size / size_of::<u32>()) }
    }

    /// The raw handle.
    #[must_use]
    pub fn as_ptr(&self) -> *mut ffi::VSGPUShader {
        self.handle.as_ptr()
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        // SAFETY: the handle is live and owned by this value.
        unsafe { (self.api.freeGPUShader)(self.handle.as_ptr()) };
    }
}

/// A compute pipeline on the core's device.
///
/// Building one turns a compiled [`Shader`] into everything a dispatch needs: a
/// shader module, a descriptor set layout with one `storage buffer` per declared
/// binding, a pipeline layout carrying the push constants, and the pipeline
/// itself. Each is created once and destroyed with this value.
///
/// Descriptors are pushed onto the command buffer rather than written into a set
/// allocated from a pool, so a pipeline owns no descriptor pool to reset between
/// frames.
#[derive(Debug)]
pub struct ComputePipeline {
    vk: ffi::VSVulkanFunctions,
    device: ffi::VkDevice,
    shader_module: ffi::VkShaderModule,
    set_layout: ffi::VkDescriptorSetLayout,
    layout: ffi::VkPipelineLayout,
    pipeline: ffi::VkPipeline,
    /// How many `storage buffer` bindings the layout declares, numbered from
    /// zero. A dispatch has to bind exactly this many buffers.
    bindings: u32,
    /// The push constant block's size in bytes, at offset zero.
    push_constant_size: u32,
}

// SAFETY: every field is a handle or a number, the handles are created on the
// core's device and immutable once built, and everything done with them is a
// Vulkan call, which the specification makes thread safe. The device outlives
// the pipeline because the core owns it.
unsafe impl Send for ComputePipeline {}
// SAFETY: as above. A pipeline holds no per thread state, which is what lets one
// instance serve every worker thread a filter is called on.
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    /// Starts building a pipeline that runs `shader`.
    #[must_use]
    pub fn builder<'a>(core: &'a Core, shader: &'a Shader) -> ComputePipelineBuilder<'a> {
        ComputePipelineBuilder {
            core,
            shader,
            bindings: 0,
            push_constant_size: 0,
        }
    }

    /// How many `storage buffer` bindings the layout declares.
    #[must_use]
    pub fn bindings(&self) -> u32 {
        self.bindings
    }

    /// The raw `VkPipeline`.
    #[must_use]
    pub fn as_ptr(&self) -> ffi::VkPipeline {
        self.pipeline
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        // SAFETY: each handle was created on this device and is given back
        // exactly once. Nothing is still using them: a pipeline only reaches the
        // GPU through a recording, and every recording is submitted or abandoned
        // before the filter instance holding this is dropped.
        unsafe {
            if self.pipeline != 0
                && let Some(destroy) = self.vk.vkDestroyPipeline
            {
                destroy(self.device, self.pipeline, null());
            }
            if self.layout != 0
                && let Some(destroy) = self.vk.vkDestroyPipelineLayout
            {
                destroy(self.device, self.layout, null());
            }
            if self.set_layout != 0
                && let Some(destroy) = self.vk.vkDestroyDescriptorSetLayout
            {
                destroy(self.device, self.set_layout, null());
            }
            if self.shader_module != 0
                && let Some(destroy) = self.vk.vkDestroyShaderModule
            {
                destroy(self.device, self.shader_module, null());
            }
        }
    }
}

/// Builds a [`ComputePipeline`].
///
/// Every [`storage_buffer()`](Self::storage_buffer) declares one binding, and the
/// order is the binding order: the first is `set = 0, binding = 0`, the second is
/// `binding = 1`, and so on. [`ExecContext::dispatch()`] binds its buffers in the
/// same order, which makes a pipeline the one place that decides what a shader
/// sees.
#[derive(Debug)]
pub struct ComputePipelineBuilder<'a> {
    core: &'a Core,
    shader: &'a Shader,
    bindings: u32,
    push_constant_size: u32,
}

impl ComputePipelineBuilder<'_> {
    /// Declares one `readonly buffer` or `writeonly buffer` binding.
    #[must_use]
    pub fn storage_buffer(mut self) -> Self {
        self.bindings = self.bindings.saturating_add(1);
        self
    }

    /// Declares a push constant block of `size` bytes at offset zero.
    ///
    /// Zero, which is the default, declares none. A size that is not a multiple
    /// of four is rounded up, since the specification requires the range to be
    /// one and every member a shader can put in the block is at least that wide.
    /// A block at another offset needs the raw path through
    /// [`ExecPool::acquire()`].
    #[must_use]
    pub fn push_constants(mut self, size: u32) -> Self {
        self.push_constant_size = size.div_ceil(4).saturating_mul(4);
        self
    }

    /// Creates the shader module, both layouts and the pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core is on no device, when it did not
    /// resolve an entry point this needs, or when the driver refused to create
    /// one of the objects. Whatever was created before the failure is destroyed.
    pub fn build(self) -> Result<ComputePipeline, VulkanError> {
        let vk = *self.core.vulkan_functions()?;
        let device = self.core.vulkan_handles()?.device;
        let mut this = ComputePipeline {
            vk,
            device,
            shader_module: 0,
            set_layout: 0,
            layout: 0,
            pipeline: 0,
            bindings: self.bindings,
            push_constant_size: self.push_constant_size,
        };

        this.create_shader_module(self.shader.code())?;
        this.create_set_layout()?;
        this.create_layout()?;
        this.create_pipeline()?;

        Ok(this)
    }
}

impl ComputePipeline {
    /// Creates the shader module from the compiled words.
    ///
    /// The first step of [`ComputePipelineBuilder::build()`] is split out from
    /// the rest so that each `unsafe` block is one call with one contract, and
    /// so a failure leaves behind exactly what `Drop` destroys.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core did not resolve
    /// `vkCreateShaderModule` or the driver refused to create the module.
    fn create_shader_module(&mut self, code: &[u32]) -> Result<(), VulkanError> {
        let info = ffi::VkShaderModuleCreateInfo {
            sType: ffi::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext: null(),
            flags: 0,
            codeSize: size_of_val(code),
            pCode: code.as_ptr(),
        };

        // SAFETY: the module is created on the core's device from the create-info
        // the specification asks for, and assigned into a field that `Drop`
        // destroys. The words outlive the call.
        unsafe {
            let create = self.vk.vkCreateShaderModule.ok_or_else(|| {
                VulkanError::new(None, "the core did not resolve vkCreateShaderModule")
            })?;

            if create(
                self.device,
                &raw const info,
                null(),
                &raw mut self.shader_module,
            ) != ffi::VK_SUCCESS
            {
                return Err(VulkanError::new(None, "could not create the shader module"));
            }
        }

        Ok(())
    }

    /// Creates the descriptor set layout: one pushed `storage buffer` binding per
    /// declared binding, numbered from zero.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core did not resolve
    /// `vkCreateDescriptorSetLayout` or the driver refused to create the layout.
    fn create_set_layout(&mut self) -> Result<(), VulkanError> {
        let bindings: Vec<ffi::VkDescriptorSetLayoutBinding> = (0..self.bindings)
            .map(|binding| ffi::VkDescriptorSetLayoutBinding {
                binding,
                descriptorType: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers: null(),
            })
            .collect();
        let info = ffi::VkDescriptorSetLayoutCreateInfo {
            sType: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext: null(),
            // Pushed onto the command buffer rather than written into a set
            // taken from a pool, which the layout has to say.
            flags: ffi::VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
            bindingCount: u32::try_from(bindings.len()).unwrap_or(0),
            pBindings: bindings.as_ptr(),
        };

        // SAFETY: as in [`Self::create_shader_module()`], and the bindings array
        // outlives the call.
        unsafe {
            let create = self.vk.vkCreateDescriptorSetLayout.ok_or_else(|| {
                VulkanError::new(None, "the core did not resolve vkCreateDescriptorSetLayout")
            })?;

            if create(
                self.device,
                &raw const info,
                null(),
                &raw mut self.set_layout,
            ) != ffi::VK_SUCCESS
            {
                return Err(VulkanError::new(
                    None,
                    "could not create the descriptor set layout",
                ));
            }
        }

        Ok(())
    }

    /// Creates the pipeline layout: set 0, and the push constant block when one
    /// was declared.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core did not resolve
    /// `vkCreatePipelineLayout` or the driver refused to create the layout.
    fn create_layout(&mut self) -> Result<(), VulkanError> {
        let set_layouts = [self.set_layout];
        let ranges = [ffi::VkPushConstantRange {
            stageFlags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: self.push_constant_size,
        }];
        let info = ffi::VkPipelineLayoutCreateInfo {
            sType: ffi::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext: null(),
            flags: 0,
            setLayoutCount: 1,
            pSetLayouts: set_layouts.as_ptr(),
            pushConstantRangeCount: u32::from(self.push_constant_size != 0),
            pPushConstantRanges: ranges.as_ptr(),
        };

        // SAFETY: as in [`Self::create_shader_module()`], and both arrays outlive
        // the call.
        unsafe {
            let create = self.vk.vkCreatePipelineLayout.ok_or_else(|| {
                VulkanError::new(None, "the core did not resolve vkCreatePipelineLayout")
            })?;

            if create(self.device, &raw const info, null(), &raw mut self.layout) != ffi::VK_SUCCESS
            {
                return Err(VulkanError::new(
                    None,
                    "could not create the pipeline layout",
                ));
            }
        }

        Ok(())
    }

    /// Creates the compute pipeline itself, with `main` as its entry point.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core did not resolve
    /// `vkCreateComputePipelines` or the driver refused to create the pipeline.
    fn create_pipeline(&mut self) -> Result<(), VulkanError> {
        let stage = ffi::VkPipelineShaderStageCreateInfo {
            sType: ffi::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext: null(),
            flags: 0,
            stage: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
            module: self.shader_module,
            pName: c"main".as_ptr(),
            pSpecializationInfo: null(),
        };
        let info = ffi::VkComputePipelineCreateInfo {
            sType: ffi::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            pNext: null(),
            flags: 0,
            stage,
            layout: self.layout,
            basePipelineHandle: 0,
            basePipelineIndex: -1,
        };

        // SAFETY: as in [`Self::create_shader_module()`]. No pipeline cache: this
        // is one pipeline, created once.
        unsafe {
            let create = self.vk.vkCreateComputePipelines.ok_or_else(|| {
                VulkanError::new(None, "the core did not resolve vkCreateComputePipelines")
            })?;

            if create(
                self.device,
                0,
                1,
                &raw const info,
                null(),
                &raw mut self.pipeline,
            ) != ffi::VK_SUCCESS
            {
                return Err(VulkanError::new(
                    None,
                    "could not create the compute pipeline",
                ));
            }
        }

        Ok(())
    }
}

/// One plane of a GPU frame, declared and ready to bind into a dispatch.
///
/// [`ExecContext::read_plane()`] and [`ExecContext::write_plane()`] hand one out,
/// which is what ties a binding to a declaration: the call that returns the
/// buffer is the same one that makes the recording keep the frame alive until
/// the submission completes.
#[derive(Clone, Copy, Debug)]
pub struct DispatchPlane {
    info: ffi::VSVulkanPlaneInfo,
}

impl DispatchPlane {
    /// The bytes the plane's buffer covers, which is its stride times its height.
    #[must_use]
    pub fn size(&self) -> ffi::VkDeviceSize {
        self.info.buffer_size
    }

    /// The raw `VkBuffer`, for a binding recorded by hand.
    #[must_use]
    pub fn buffer(&self) -> ffi::VkBuffer {
        self.info.buffer
    }
}

/// A scratch buffer from the core's memory pool.
///
/// This comes from the same sub allocator frame planes do, so it is counted
/// against the VRAM limit, predicted by the admission control of the thread pool
/// and recycled through the size buckets; allocating and destroying one per
/// frame is cheap by design. Host visible requests come back persistently
/// mapped.
#[derive(Debug)]
pub struct GpuBuffer {
    handle: NonNull<ffi::VSGPUBuffer>,
    api: VsVulkanApi,
    info: ffi::VSVulkanBufferInfo,
}

// SAFETY: the buffer is owned memory from the core's pool, not tied to a thread.
unsafe impl Send for GpuBuffer {}

impl GpuBuffer {
    /// Everything a kernel or the host needs to use the buffer.
    #[must_use]
    pub fn info(&self) -> &ffi::VSVulkanBufferInfo {
        &self.info
    }

    /// The raw `VkBuffer`.
    #[must_use]
    pub fn buffer(&self) -> ffi::VkBuffer {
        self.info.buffer
    }

    /// The buffer's device address.
    #[must_use]
    pub fn address(&self) -> ffi::VkDeviceAddress {
        self.info.address
    }

    /// The buffer's size in bytes.
    #[must_use]
    pub fn size(&self) -> ffi::VkDeviceSize {
        self.info.size
    }

    /// The memory property flags the allocation actually got.
    #[must_use]
    pub fn memory_flags(&self) -> ffi::VkMemoryPropertyFlags {
        self.info.memory_flags
    }

    /// The raw handle.
    #[must_use]
    pub fn as_ptr(&self) -> *mut ffi::VSGPUBuffer {
        self.handle.as_ptr()
    }

    /// The host mapping, when the allocation is host visible.
    ///
    /// # Safety
    ///
    /// The returned slice aliases memory the GPU may be reading or writing
    /// concurrently; the caller must ensure no submission that uses the buffer is
    /// in flight for as long as the borrow lasts.
    #[must_use]
    pub unsafe fn mapped_slice(&mut self) -> Option<&mut [u8]> {
        if self.info.mapped.is_null() {
            return None;
        }

        let size = usize::try_from(self.info.size).ok()?;

        // SAFETY: a host visible allocation is mapped for `size` bytes, and the
        // caller promises nothing else touches them.
        Some(unsafe { slice::from_raw_parts_mut(self.info.mapped.cast::<u8>(), size) })
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // SAFETY: the handle is live and owned by this value. A buffer a
        // submission still uses would only have been handed over by an `unsafe`
        // call that promised the submission had completed.
        unsafe { (self.api.destroyGPUBuffer)(self.handle.as_ptr()) };
    }
}

/// Implements identity comparison for a handle owning type whose cached
/// description of the object cannot be compared, as the `info` structs a buffer
/// and a region carry cannot.
macro_rules! handle_identity {
    ($($type:ty),+ $(,)?) => {
        $(
            impl PartialEq for $type {
                fn eq(&self, other: &Self) -> bool {
                    self.handle == other.handle
                }
            }

            impl Eq for $type {}

            impl Hash for $type {
                fn hash<H: Hasher>(&self, state: &mut H) {
                    self.handle.hash(state);
                }
            }
        )+
    };
}

/// A region of the pool that frame planes come from, for a resource this API has
/// no constructor for.
///
/// Create the resource, ask Vulkan what memory it needs, allocate it here, and
/// bind against the memory and offset that come back. Binding at
/// [`GpuMemory::offset()`] rather than at the start of [`GpuMemory::memory()`]
/// matters: the region sits inside a shared block, and images are aligned
/// further inside it since their requirements are routinely coarser than the
/// pool's regions.
///
/// A freed region returns to the pool at once and may back another resource
/// immediately, so freeing early aliases live data rather than faulting.
#[derive(Debug)]
pub struct GpuMemory {
    handle: NonNull<ffi::VSGPUMemory>,
    api: VsVulkanApi,
    info: ffi::VSVulkanMemoryInfo,
}

// SAFETY: the region is owned memory from the core's pool, not tied to a thread.
unsafe impl Send for GpuMemory {}

impl GpuMemory {
    /// Everything a kernel or the host needs to use the region.
    #[must_use]
    pub fn info(&self) -> &ffi::VSVulkanMemoryInfo {
        &self.info
    }

    /// The raw `VkDeviceMemory`.
    #[must_use]
    pub fn memory(&self) -> ffi::VkDeviceMemory {
        self.info.memory
    }

    /// The byte offset inside [`Self::memory()`] to bind at.
    #[must_use]
    pub fn offset(&self) -> ffi::VkDeviceSize {
        self.info.offset
    }

    /// The region's size in bytes.
    #[must_use]
    pub fn size(&self) -> ffi::VkDeviceSize {
        self.info.size
    }

    /// The memory property flags the allocation actually got.
    #[must_use]
    pub fn memory_flags(&self) -> ffi::VkMemoryPropertyFlags {
        self.info.memory_flags
    }

    /// The raw handle.
    #[must_use]
    pub fn as_ptr(&self) -> *mut ffi::VSGPUMemory {
        self.handle.as_ptr()
    }
}

impl Drop for GpuMemory {
    fn drop(&mut self) {
        // SAFETY: the handle is live and owned by this value, and the caller has
        // promised that nothing still uses the region.
        unsafe { (self.api.freeGPUMemory)(self.handle.as_ptr()) };
    }
}

handle_identity!(GpuBuffer, GpuMemory);

/// A declaration of GPU memory the core did not allocate, so the byte count
/// takes part in its budgeting.
///
/// This holds no memory itself, only a number. Declaring a foreign allocation
/// is how the frame cache, the admission control and the unified memory brake
/// come to see it; the core cannot veto memory it does not own, so a reservation
/// is never refused.
///
/// `bytes` is an absolute total rather than a delta, so publishing what you hold
/// makes drift impossible however the calls interleave.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GpuMemoryReservation {
    handle: NonNull<ffi::VSGPUMemoryReservation>,
    api: VsVulkanApi,
}

// SAFETY: the reservation is a counter on the core, not tied to a thread.
unsafe impl Send for GpuMemoryReservation {}

impl GpuMemoryReservation {
    /// Publishes the absolute total this reservation stands for.
    ///
    /// Increasing it can evict cached GPU frames and hand idle allocator blocks
    /// back to the driver before the call returns, so reserve or update *before*
    /// the foreign allocation and the VRAM has actually been vacated. Shrinking
    /// just subtracts, and a negative total counts as zero.
    pub fn update(&mut self, bytes: i64) {
        // SAFETY: the handle is live for as long as `self`.
        unsafe { (self.api.updateGPUMemoryReservation)(self.handle.as_ptr(), bytes) };
    }

    /// The raw handle.
    #[must_use]
    pub fn as_ptr(&self) -> *mut ffi::VSGPUMemoryReservation {
        self.handle.as_ptr()
    }
}

impl Drop for GpuMemoryReservation {
    fn drop(&mut self) {
        // SAFETY: the handle is live and owned by this value, and no other call
        // on it can overlap this one because the reservation was moved here.
        unsafe { (self.api.releaseGPUMemoryReservation)(self.handle.as_ptr()) };
    }
}

/// A pool of command buffers and their timeline, for recording your own work.
///
/// A pool belongs to one filter instance and is created on one of the core's
/// queues: a pool on [`Transfer`](ffi::VSVulkanQueueType::Transfer) may only ever
/// record copies, and a pool anything is dispatched into belongs on
/// [`Compute`](ffi::VSVulkanQueueType::Compute).
///
/// The core sizes the pool's context ring itself, from its worker thread count,
/// and bounds how much memory queued submissions may pin separately. Acquiring
/// waits out the ring's oldest submission and may additionally wait on the
/// core's device wide in-flight budget, so a graph running far ahead of the GPU
/// shows up as an occasionally slower acquire and nothing else.
///
/// Shared across a filter's worker threads, which is what the pool is for: one
/// context per thread at a time.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ExecPool {
    handle: NonNull<ffi::VSGPUExecPool>,
    api: VsVulkanApi,
}

// SAFETY: the pool is designed to be handed to every worker thread of a filter;
// each acquires its own context, and the ring is internally synchronized.
unsafe impl Send for ExecPool {}
// SAFETY: as above. The waiting calls stay `unsafe` because of what they do to
// the calling thread's own state, not because of data races.
unsafe impl Sync for ExecPool {}

// Set while this thread is inside `ExecPool::record()`.
//
// The one context per thread rule is the whole reason `ExecPool::acquire()`
// is `unsafe`, and a recording that cannot outlive a call is exactly what makes
// it checkable, so it is checked here instead of documented.
std::thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

/// Clears this thread's recording claim again, however `record` ends.
struct RecordingClaim;

impl RecordingClaim {
    /// Claims the calling thread, or reports that it already holds a recording.
    fn claim() -> Result<Self, VulkanError> {
        RECORDING.with(|recording| {
            if recording.replace(true) {
                return Err(VulkanError::new(
                    None,
                    "this thread already holds an exec context",
                ));
            }

            Ok(RecordingClaim)
        })
    }
}

impl Drop for RecordingClaim {
    fn drop(&mut self) {
        RECORDING.with(|recording| recording.set(false));
    }
}

impl ExecPool {
    /// The pool's own timeline.
    ///
    /// The pool holds its own reference, so publishing values on it needs no
    /// reference of yours. Never signal it yourself: the pool allocates values
    /// inside the queue lock so signals reach the queue in increasing order, and
    /// [`ExecContext::submit()`] is what produces a value you may wait on.
    #[must_use]
    pub fn timeline(&self) -> TimelineRef {
        // SAFETY: the handle is live for as long as `self`, and the pool keeps
        // the timeline alive for its own lifetime.
        TimelineRef(unsafe { (self.api.gpuExecPoolTimeline)(self.handle.as_ptr()) })
    }

    /// The raw handle.
    #[must_use]
    pub fn as_ptr(&self) -> *mut ffi::VSGPUExecPool {
        self.handle.as_ptr()
    }

    /// Claims a recording slot for this thread.
    ///
    /// The returned context is one recording, owned by the calling thread until
    /// it is submitted or abandoned, and neither call needs to be on the same
    /// thread as the acquire.
    ///
    /// # Safety
    ///
    /// One context per thread, of this or any other pool: a ring has only two to
    /// eight contexts, so a second acquire from a thread already holding one
    /// could only wait for a slot nobody would ever release. That case is fatal
    /// rather than merely wrong, reported when the second acquire would wait.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if no context could be claimed.
    pub unsafe fn acquire(&self) -> Result<ExecContext<'_>, VulkanError> {
        let (handle, message) = with_error_buffer(|buffer, size| unsafe {
            (self.api.gpuExecAcquire)(self.handle.as_ptr(), buffer, size)
        });

        NonNull::new(handle)
            .map(|handle| ExecContext {
                handle,
                api: self.api,
                _pool: PhantomData,
                declared: [null(); DECLARED_READ_SLOTS],
            })
            .ok_or_else(|| VulkanError::new(message, "could not acquire an exec context"))
    }

    /// Waits out the whole pool and runs every pending release callback.
    ///
    /// Prefer [`ExecContext::submit()`]'s value with [`ExecPool::wait_value()`]
    /// when only your own submission matters: this waits the pool's newest
    /// submission, so concurrent frames serialize on each other's work.
    ///
    /// # Safety
    ///
    /// The calling thread must not hold an exec context, of this or any other
    /// pool; doing so is fatal.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the wait could not be established, or if the
    /// library reported a drain result outside the documented set.
    pub unsafe fn wait_idle(&self) -> Result<ffi::VSGPUDrainResult, VulkanError> {
        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (self.api.gpuExecPoolWaitIdle)(self.handle.as_ptr(), buffer, size)
        });

        drain_result(result).ok_or_else(|| {
            VulkanError::new(message, "draining an exec pool reported an unknown result")
        })
    }

    /// Waits for one value on the pool's timeline.
    ///
    /// This is the same wait as [`Timeline::wait_value()`] on
    /// [`Self::timeline()`], with the value check the pool can make and a
    /// timeline you signal yourself cannot.
    ///
    /// # Safety
    ///
    /// The value must be one this pool submitted: a
    /// [`ExecContext::submit()`] result, or anything earlier. A value past the
    /// newest submission is fatal, since no consumer would ever signal it.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the wait could not be established, or if the
    /// library reported a drain result outside the documented set.
    pub unsafe fn wait_value(&self, value: u64) -> Result<ffi::VSGPUDrainResult, VulkanError> {
        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (self.api.gpuExecWaitValue)(self.handle.as_ptr(), value, buffer, size)
        });

        drain_result(result).ok_or_else(|| {
            VulkanError::new(
                message,
                "waiting on an exec pool reported an unknown result",
            )
        })
    }

    /// Runs `f` against one recording and submits it.
    ///
    /// This is the safe way to record: the context is claimed for this thread,
    /// handed to `f`, and submitted when `f` returns without an error. It cannot
    /// outlive the call, so the one context per thread rule that makes
    /// [`ExecPool::acquire()`] `unsafe` is checked here, and a nested call is
    /// refused rather than being fatal.
    ///
    /// The returned value is the submission's point on
    /// [`ExecPool::timeline()`], which is what [`ExecPool::wait_value()`] takes.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if this thread already holds a recording, if no
    /// context could be claimed, if `f` returned one, in which case the recording
    /// is abandoned rather than submitted, or if the submission failed.
    pub fn record<F>(&self, f: F) -> Result<u64, VulkanError>
    where
        F: FnOnce(&mut ExecContext<'_>) -> Result<(), VulkanError>,
    {
        let _claim = RecordingClaim::claim()?;

        // SAFETY: the claim proves this thread holds no other recording, which
        // is all `acquire` asks of its caller.
        let mut context = unsafe { self.acquire()? };

        if let Err(error) = f(&mut context) {
            context.abandon();
            return Err(error);
        }

        // SAFETY: the recording is submitted from the thread that claimed it, and
        // every declaration made against it came from this API: frames were
        // declared by the calls that handed out their planes, and nothing was
        // handed over by hand.
        unsafe { context.submit() }
    }
}

impl Drop for ExecPool {
    fn drop(&mut self) {
        // SAFETY: the handle is live and owned by this value, and no context can
        // outlive the pool because `ExecContext` borrows it.
        unsafe { (self.api.freeGPUExecPool)(self.handle.as_ptr()) };
    }
}

/// How many frames one recording remembers having declared a read for.
///
/// Only discovery is capped: past the last slot a frame is declared again, which
/// costs accounting in the core but is never wrong.
const DECLARED_READ_SLOTS: usize = 8;

/// One recording of an [`ExecPool`], claimed by one thread.
///
/// The context is consumed by [`ExecContext::submit()`], [`ExecContext::abandon()`]
/// or by dropping it, which abandons the recording. It is not `Send`: recording,
/// retaining, submitting and abandoning all have to happen on the thread that
/// acquired it.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ExecContext<'pool> {
    handle: NonNull<ffi::VSGPUExecContext>,
    api: VsVulkanApi,
    /// Keeps the pool alive for as long as a recording is outstanding, and keeps
    /// the context off other threads.
    _pool: PhantomData<&'pool ExecPool>,
    /// The frames this recording has already declared a read for.
    ///
    /// A frame is declared once: the core takes a reference and meters its bytes
    /// on every call, and the waits it contributes are the same each time, so
    /// declaring it per plane would only count it per plane.
    declared: [*const ffi::VSFrame; DECLARED_READ_SLOTS],
}

impl ExecContext<'_> {
    /// The command buffer being recorded.
    ///
    /// Put anything Vulkan allows into it, then submit.
    #[must_use]
    pub fn command_buffer(&self) -> ffi::VkCommandBuffer {
        // SAFETY: the context is live until it is submitted or abandoned.
        unsafe { (self.api.gpuExecCommandBuffer)(self.handle.as_ptr()) }
    }

    /// The raw handle.
    ///
    /// Do not keep it past the call that ends the recording: the handle lives in
    /// its ring slot for the life of the pool, so a kept pointer is
    /// indistinguishable from a later, unrelated recording on the same slot.
    #[must_use]
    pub fn as_ptr(&self) -> *mut ffi::VSGPUExecContext {
        self.handle.as_ptr()
    }

    /// Declares that this submission reads `frame`.
    ///
    /// Every plane's producer pair becomes a device side wait, and the frame is
    /// kept alive until the submission completes. Declaring the same frame again
    /// in one recording is allowed and settles nothing twice, which is what lets
    /// [`ExecContext::read_plane()`] be called once per plane.
    ///
    /// A plane a foreign API holds is refused by the core rather than recorded,
    /// so this is not a memory error to make: the core owns the check.
    pub fn read(&mut self, frame: &VideoFrame) {
        let ptr = frame.as_ptr().cast_const();
        if self.declared.contains(&ptr) {
            return;
        }

        if let Some(slot) = self.declared.iter_mut().find(|slot| slot.is_null()) {
            *slot = ptr;
        }

        // SAFETY: the recording is live, the frame is borrowed for the call, and
        // the context takes its own reference, so the frame's lifetime from here
        // is the core's business.
        unsafe { self.reads_frame(frame) };
    }

    /// Declares a read of `frame` and hands back the plane to bind.
    ///
    /// This is [`ExecContext::read()`] and the buffer a dispatch needs in one
    /// call, which is what keeps a recording from binding a plane it never
    /// declared.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the frame is not GPU resident or the plane
    /// does not exist.
    pub fn read_plane(
        &mut self,
        frame: &VideoFrame,
        plane: i32,
    ) -> Result<DispatchPlane, VulkanError> {
        let Some(info) = frame.gpu_plane(plane) else {
            return Err(VulkanError::new(
                None,
                "the frame is not GPU resident, or has no plane there",
            ));
        };
        self.read(frame);

        Ok(DispatchPlane { info })
    }

    /// Declares that this submission writes `plane` and hands it back to bind.
    ///
    /// Submitting publishes the pool's `(timeline, value)` on the plane as its
    /// producer pair, which is how every later consumer knows when it is ready
    /// without waiting for anything.
    ///
    /// A plane the frame shares with another frame is refused by the core rather
    /// than written: a GPU plane has no copy on write, so the write would land in
    /// both.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the frame is not GPU resident or the plane
    /// does not exist.
    pub fn write_plane(
        &mut self,
        frame: &mut VideoFrame,
        plane: i32,
    ) -> Result<DispatchPlane, VulkanError> {
        let Some(info) = frame.gpu_plane(plane) else {
            return Err(VulkanError::new(
                None,
                "the frame is not GPU resident, or has no plane there",
            ));
        };

        // SAFETY: the recording is live, the frame is borrowed mutably for the
        // call, and the plane is the caller's to declare.
        unsafe { self.writes_plane(frame, plane) };

        Ok(DispatchPlane { info })
    }

    /// Records one compute dispatch into the recording.
    ///
    /// `buffers` are bound to the pipeline's bindings in order, and there have to
    /// be exactly as many as the pipeline declared: a shader reading a binding
    /// nothing was written to reads whatever the driver left there, so a short
    /// list is refused instead of recorded.
    ///
    /// `push_constants` is the block at offset zero and must fit the layout.
    /// `groups` is the workgroup count, so the invocations are that times the
    /// shader's `local_size`. Nothing waits for the dispatch here: the
    /// submission's timeline value is what says when it has run.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the buffer count or the push constant block
    /// does not match the pipeline, or when the core did not resolve an entry
    /// point this needs.
    pub fn dispatch(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[DispatchPlane],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<(), VulkanError> {
        if u32::try_from(buffers.len()).unwrap_or(u32::MAX) != pipeline.bindings {
            return Err(VulkanError::new(
                None,
                "the dispatch binds a different number of buffers than the pipeline declares",
            ));
        }
        let size = u32::try_from(push_constants.len()).unwrap_or(u32::MAX);
        if size % 4 != 0 {
            return Err(VulkanError::new(
                None,
                "the push constants are not a whole number of 32 bit values",
            ));
        }
        if size > pipeline.push_constant_size {
            return Err(VulkanError::new(
                None,
                "the push constants do not fit the pipeline layout",
            ));
        }

        let (bind, push_descriptors, push_constants_call, dispatch_call) = (
            pipeline.vk.vkCmdBindPipeline.ok_or_else(|| {
                VulkanError::new(None, "the core did not resolve vkCmdBindPipeline")
            })?,
            pipeline.vk.vkCmdPushDescriptorSet.ok_or_else(|| {
                VulkanError::new(None, "the core did not resolve vkCmdPushDescriptorSet")
            })?,
            pipeline.vk.vkCmdPushConstants2.ok_or_else(|| {
                VulkanError::new(None, "the core did not resolve vkCmdPushConstants2")
            })?,
            pipeline
                .vk
                .vkCmdDispatch
                .ok_or_else(|| VulkanError::new(None, "the core did not resolve vkCmdDispatch"))?,
        );

        let infos: Vec<ffi::VkDescriptorBufferInfo> = buffers
            .iter()
            .map(|plane| ffi::VkDescriptorBufferInfo {
                buffer: plane.info.buffer,
                offset: 0,
                range: plane.info.buffer_size,
            })
            .collect();
        let writes: Vec<ffi::VkWriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| ffi::VkWriteDescriptorSet {
                sType: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: null(),
                // Push descriptors have no set to write into; the field is
                // ignored, and the binding lands on the layout's set 0.
                dstSet: 0,
                dstBinding: u32::try_from(binding).unwrap_or(0),
                dstArrayElement: 0,
                descriptorCount: 1,
                descriptorType: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo: null(),
                pBufferInfo: info,
                pTexelBufferView: null(),
            })
            .collect();
        let constants = ffi::VkPushConstantsInfo {
            sType: ffi::VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
            pNext: null(),
            layout: pipeline.layout,
            stageFlags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: u32::try_from(push_constants.len()).unwrap_or(0),
            pValues: push_constants.as_ptr().cast(),
        };
        let cb = self.command_buffer();

        // SAFETY: the recording is live and is the only one this thread holds.
        // Every buffer bound here was handed out by `read_plane` or `write_plane`
        // and so was declared on this recording, which is what keeps the frames
        // behind them alive until the submission completes. The push constant
        // bytes and the descriptor infos are only read during the call.
        unsafe {
            bind(cb, ffi::VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            if !push_constants.is_empty() {
                push_constants_call(cb, &raw const constants);
            }
            if !writes.is_empty() {
                push_descriptors(
                    cb,
                    ffi::VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline.layout,
                    0,
                    u32::try_from(writes.len()).unwrap_or(0),
                    writes.as_ptr(),
                );
            }
            dispatch_call(cb, groups[0], groups[1], groups[2]);
        }

        Ok(())
    }

    /// Declares that this submission reads the frame.
    ///
    /// The frame's planes' producer pairs become device side waits, and the
    /// frame is kept alive until the submission completes. The context takes its
    /// own reference, so the caller still releases its own normally.
    ///
    /// # Safety
    ///
    /// Fatal on a frame with a plane handed to a foreign API, since taking such a
    /// plane back is the foreign side's completion to publish, not this
    /// submission's to wait on.
    pub unsafe fn reads_frame(&self, frame: &VideoFrame) {
        // SAFETY: the context is live, and the frame outlives the call. The
        // context takes its own reference.
        unsafe { (self.api.gpuExecReadsFrame)(self.handle.as_ptr(), frame.as_ptr().cast_const()) };
    }

    /// Declares that this submission writes the plane.
    ///
    /// Submitting publishes the pool's `(timeline, value)` on the plane as its
    /// producer pair, and the context takes its own reference to the frame,
    /// dropped once the pair is published.
    ///
    /// # Safety
    ///
    /// The frame must own the plane outright. Copying a frame shares its planes,
    /// and a GPU plane has no copy on write, so a write declared on a shared
    /// plane would land in the other frame as well; that is fatal rather than
    /// silent. Fatal on a plane handed to a foreign API too.
    pub unsafe fn writes_plane(&self, frame: &mut VideoFrame, plane: i32) {
        // SAFETY: the context is live, the frame is borrowed for the call, and
        // the plane is the caller's to declare.
        unsafe { (self.api.gpuExecWritesPlane)(self.handle.as_ptr(), frame.as_ptr(), plane) };
    }

    /// Hands a scratch buffer to the context.
    ///
    /// Ownership transfers, and the context destroys the buffer once the
    /// submission completes, so it must not be destroyed by hand afterwards.
    ///
    /// # Safety
    ///
    /// Surrendering a [`GpuBuffer`] means its `Drop` never runs, so the caller
    /// must give up its ownership for real, with [`std::mem::forget`] or
    /// by holding it in an [`Option`] that is taken.
    pub unsafe fn uses_buffer(&self, buffer: GpuBuffer) {
        let ptr = buffer.as_ptr();
        forget(buffer);

        // SAFETY: the context is live and takes ownership of the handle.
        unsafe { (self.api.gpuExecUsesBuffer)(self.handle.as_ptr(), ptr) };
    }

    /// Hands a region of pool memory to the context.
    ///
    /// The region stays out of the pool until the submission being recorded has
    /// completed, and its bytes count against the in-flight retention budget
    /// meanwhile. Ownership transfers exactly as it does for
    /// [`ExecContext::uses_buffer()`], and the resource bound to the region is
    /// still the caller's to destroy.
    ///
    /// # Safety
    ///
    /// As [`ExecContext::uses_buffer()`]: the handle is consumed, so it must be
    /// surrendered for real.
    pub unsafe fn uses_memory(&self, memory: GpuMemory) {
        let ptr = memory.as_ptr();
        forget(memory);

        // SAFETY: the context is live and takes ownership of the handle.
        unsafe { (self.api.gpuExecUsesMemory)(self.handle.as_ptr(), ptr) };
    }

    /// Registers cleanup of your own, on the context's retention list.
    ///
    /// This is for what the typed calls cannot name, a per frame image being the
    /// case: the image, every view recorded against it and the region underneath
    /// have to retire together, and one registration of a struct holding all
    /// three frees them in the right order where a call per object could not
    /// express the ordering.
    ///
    /// `bytes` is what the object pins in device memory, counted against the
    /// in-flight retention budget from submit until release. Pass zero for host
    /// side bookkeeping, and do not double count bytes a typed call already
    /// counted for the same object.
    ///
    /// # Safety
    ///
    /// `release` runs on whichever thread reaps the submission, so it must be
    /// thread safe and must not do what a release callback is forbidden from
    /// doing: acquiring a context, allocating GPU memory, or creating, freeing or
    /// waiting on a pool from inside it are all fatal. It may take its own locks
    /// and free whatever the object owns, and that is all. `object` must stay
    /// valid until the callback runs.
    pub unsafe fn retain(
        &self,
        release: ffi::VSGPUReleaseFunc,
        object: *mut c_void,
        bytes: ffi::VkDeviceSize,
    ) {
        // SAFETY: the contract is the caller's, documented above and in the C API.
        unsafe { (self.api.gpuExecRetain)(self.handle.as_ptr(), release, object, bytes) };
    }

    /// Gives up the recording without submitting.
    ///
    /// This is what dropping the context does; the method exists to make the
    /// intent explicit at the call site.
    pub fn abandon(self) {
        drop(self);
    }

    /// Ends recording and submits, publishing the producer pairs.
    ///
    /// The timeline value is allocated inside the queue lock so that signals
    /// reach the queue in increasing order. The returned value is what this
    /// submission signals on the pool's timeline; pass it to
    /// [`ExecPool::wait_value()`] to wait for exactly this submission.
    ///
    /// Do not wait on the timeline with `vkWaitSemaphores` yourself: a bare wait
    /// cannot distinguish a completed dispatch from a reset that force-signalled
    /// the timeline, and reports the second as the first.
    ///
    /// # Safety
    ///
    /// The context must be submitted from the thread that acquired it. Every
    /// declaration made against it must hold: frames it wrote must own their
    /// planes, buffers and regions it took ownership of must not be dropped by
    /// hand, and every retain callback must satisfy the contract of
    /// [`ExecContext::retain()`].
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the submission failed. The context is consumed
    /// either way, as it is in the C API.
    pub unsafe fn submit(self) -> Result<u64, VulkanError> {
        let api = self.api;
        let handle = self.handle.as_ptr();
        let mut value = 0u64;

        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (api.gpuExecSubmit)(handle, &raw mut value, buffer, size)
        });

        // The context is consumed whether the submission succeeded or not, so
        // the `Drop` that would abandon it must not run.
        forget(self);

        if result != 0 {
            return Err(VulkanError::new(message, "could not submit a recording"));
        }

        Ok(value)
    }
}

impl Drop for ExecContext<'_> {
    fn drop(&mut self) {
        // A recording that was neither submitted nor abandoned would pin its
        // ring slot forever, so give it back.
        //
        // SAFETY: the context is live and owned by this value.
        unsafe { (self.api.gpuExecAbandon)(self.handle.as_ptr()) };
    }
}

impl Core {
    /// The GPU API table.
    #[must_use]
    pub fn vulkan(&self) -> VsVulkanApi {
        self.api().vulkan()
    }

    /// Selects the physical device this core runs on.
    ///
    /// `index` is one of [`VsVulkanApi::enumerate_devices()`], and the call
    /// fails if the entry is unusable. Setting a device also sets the core's
    /// default VRAM limit, which [`Core::set_max_vram_use()`] can then move.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the index is out of range or the device is not
    /// usable.
    pub fn set_vulkan_device(&self, index: i32) -> Result<(), VulkanError> {
        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (self.vulkan().setVulkanDevice)(self.as_ptr(), index, buffer, size)
        });

        if result != 0 {
            return Err(VulkanError::new(
                message,
                "could not select a Vulkan device",
            ));
        }

        Ok(())
    }

    /// What the core knows about the device it is on.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the core has no GPU device.
    pub fn vulkan_core_info(&self) -> Result<VulkanCoreInfo, VulkanError> {
        let api = self.vulkan();
        let mut info = MaybeUninit::<ffi::VSVulkanCoreInfo>::uninit();

        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (api.getVulkanCoreInfo)(self.as_ptr(), info.as_mut_ptr(), buffer, size)
        });

        if result != 0 {
            return Err(VulkanError::new(message, "the core has no Vulkan device"));
        }

        // SAFETY: the call reported success, so every field was written.
        Ok(VulkanCoreInfo::from(unsafe { info.assume_init_ref() }))
    }

    /// Sets how much VRAM the core's allocator may use, and returns the old
    /// limit.
    ///
    pub fn set_max_vram_use(&mut self, bytes: i64) -> i64 {
        // SAFETY: the core is live, and the call only adjusts a limit.
        unsafe { (self.vulkan().setMaxVRAMUse)(bytes, self.as_ptr()) }
    }

    /// Everything needed to run your own Vulkan work on the core's device.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the core has no GPU device.
    pub fn vulkan_handles(&self) -> Result<ffi::VSVulkanCoreHandles, VulkanError> {
        let api = self.vulkan();
        let mut handles = MaybeUninit::<ffi::VSVulkanCoreHandles>::uninit();

        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (api.getVulkanHandles)(self.as_ptr(), handles.as_mut_ptr(), buffer, size)
        });

        if result != 0 {
            return Err(VulkanError::new(message, "the core has no Vulkan device"));
        }

        // SAFETY: the call reported success, so every field was written.
        Ok(unsafe { handles.assume_init() })
    }

    /// The Vulkan entry points the core resolved for this device.
    ///
    /// These are the same functions the core uses, resolved through its instance
    /// and device, so dispatching through them needs no loader of your own.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the core has no GPU device.
    pub fn vulkan_functions(&self) -> Result<&ffi::VSVulkanFunctions, VulkanError> {
        let (functions, message) = with_error_buffer(|buffer, size| unsafe {
            (self.vulkan().getVulkanFunctions)(self.as_ptr(), buffer, size)
        });

        // SAFETY: the table belongs to the core, which outlives the borrow, and
        // it is one struct of function pointers with no interior state.
        (!functions.is_null())
            .then(|| unsafe { &*functions })
            .ok_or_else(|| VulkanError::new(message, "the core has no Vulkan device"))
    }

    /// Creates a GPU resident video frame.
    ///
    /// The frame behaves exactly like a CPU one everywhere else: same
    /// properties, same dimensions, same `std.FrameEval` and `std.ModifyFrame`
    /// support. Only the dimensions and the format are fixed at creation, and
    /// `prop_src` supplies the properties, as it does for
    /// [`Core::new_video_frame()`].
    ///
    /// Returns [`None`] when the core has no GPU device, which is not the same
    /// as an allocation failure: the core reports that one by returning a frame
    /// with no planes, as it does for CPU frames.
    #[must_use]
    pub fn new_gpu_video_frame(
        &self,
        format: &VideoFormat,
        width: i32,
        height: i32,
        prop_src: Option<&VideoFrame>,
    ) -> Option<VideoFrame> {
        let api = self.vulkan();
        let src = prop_src.map_or(null(), |frame| frame.as_ptr().cast_const());

        // SAFETY: the core is live, the format outlives the call, and the frame
        // created belongs to the caller.
        let ptr = unsafe { (api.newGPUVideoFrame)(format, width, height, src, self.as_ptr()) };

        (!ptr.is_null()).then(|| unsafe { VideoFrame::from_ptr(ptr, self.api()) })
    }

    /// Creates a timeline of your own, for recording and submitting without the
    /// core's exec pool.
    ///
    /// The timeline starts at zero and is returned with one reference, which is
    /// yours to release. Release it without waiting for consumers: planes you
    /// published it on hold their own.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the timeline could not be created.
    pub fn create_gpu_timeline(&self) -> Result<Timeline, VulkanError> {
        let api = self.vulkan();
        let (handle, message) = with_error_buffer(|buffer, size| unsafe {
            (api.createGPUTimeline)(self.as_ptr(), buffer, size)
        });

        Timeline::from_owned(handle, api)
            .ok_or_else(|| VulkanError::new(message, "could not create a GPU timeline"))
    }

    /// Allocates a scratch buffer from the core's pool.
    ///
    /// Host visible requests come back persistently mapped, which is what
    /// [`GpuBuffer::mapped_slice()`] hands out.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the allocation failed.
    pub fn create_gpu_buffer(
        &self,
        size: ffi::VkDeviceSize,
        usage: ffi::VkBufferUsageFlags,
        required_flags: ffi::VkMemoryPropertyFlags,
        preferred_flags: ffi::VkMemoryPropertyFlags,
    ) -> Result<GpuBuffer, VulkanError> {
        let api = self.vulkan();
        let mut info = MaybeUninit::<ffi::VSVulkanBufferInfo>::uninit();

        let (handle, message) = with_error_buffer(|buffer, size_of_message| unsafe {
            (api.createGPUBuffer)(
                self.as_ptr(),
                size,
                usage,
                required_flags,
                preferred_flags,
                info.as_mut_ptr(),
                buffer,
                size_of_message,
            )
        });

        NonNull::new(handle)
            .map(|handle| GpuBuffer {
                handle,
                api,
                // SAFETY: the call reported success, so every field was written.
                info: unsafe { info.assume_init() },
            })
            .ok_or_else(|| VulkanError::new(message, "could not allocate a GPU buffer"))
    }

    /// Allocates a region of the pool for a resource this API has no
    /// constructor for.
    ///
    /// Bind at [`GpuMemory::offset()`], not at the start of the memory, and
    /// check the two things the requirements cannot tell you: the resource must
    /// not be created with external memory info, since these blocks are not
    /// exportable, and its `VkMemoryDedicatedRequirements` must not report
    /// `requiresDedicatedAllocation`.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the allocation failed.
    pub fn allocate_gpu_memory(
        &self,
        requirements: &ffi::VkMemoryRequirements,
        required_flags: ffi::VkMemoryPropertyFlags,
        preferred_flags: ffi::VkMemoryPropertyFlags,
    ) -> Result<GpuMemory, VulkanError> {
        let api = self.vulkan();
        let mut info = MaybeUninit::<ffi::VSVulkanMemoryInfo>::uninit();

        let (handle, message) = with_error_buffer(|buffer, size| unsafe {
            (api.allocateGPUMemory)(
                self.as_ptr(),
                requirements,
                required_flags,
                preferred_flags,
                info.as_mut_ptr(),
                buffer,
                size,
            )
        });

        NonNull::new(handle)
            .map(|handle| GpuMemory {
                handle,
                api,
                // SAFETY: the call reported success, so every field was written.
                info: unsafe { info.assume_init() },
            })
            .ok_or_else(|| VulkanError::new(message, "could not allocate GPU memory"))
    }

    /// Declares GPU memory the core did not allocate into its accounting.
    ///
    /// Only declare memory on the core's own device, matching
    /// [`VulkanCoreInfo::device_uuid`] or [`VulkanCoreInfo::device_luid`], and
    /// never declare bytes the core already accounts for, or they count twice.
    /// The exception is imports of planes [`VideoFrame::export_gpu_plane()`]
    /// handed out, which belong in the declaration where
    /// [`ffi::VS_GPU_IMPORT_CHARGED_AGAIN`] is 1.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the reservation could not be created.
    pub fn reserve_gpu_memory(&self, bytes: i64) -> Result<GpuMemoryReservation, VulkanError> {
        let api = self.vulkan();
        let (handle, message) = with_error_buffer(|buffer, size| unsafe {
            (api.reserveGPUMemory)(self.as_ptr(), bytes, buffer, size)
        });

        NonNull::new(handle)
            .map(|handle| GpuMemoryReservation { handle, api })
            .ok_or_else(|| VulkanError::new(message, "could not reserve GPU memory"))
    }

    /// Compiles a compute shader at runtime.
    ///
    /// Results are cached per core by source text, so repeating a kernel from
    /// many filter instances parses it once. Specialize by concatenating a
    /// `#define` preamble in front of the kernel body; there is no include
    /// handler.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] with the compiler log if the source did not
    /// compile, including for a language the core does not know.
    pub fn compile_gpu_shader(
        &self,
        language: ffi::VSGPUShaderLanguage,
        source: &CStr,
    ) -> Result<Shader, VulkanError> {
        let api = self.vulkan();
        let (handle, message) = with_error_buffer(|buffer, size| unsafe {
            (api.compileGPUShader)(
                self.as_ptr(),
                language as i32,
                source.as_ptr(),
                buffer,
                size,
            )
        });

        NonNull::new(handle)
            .map(|handle| Shader { handle, api })
            .ok_or_else(|| VulkanError::new(message, "could not compile the shader"))
    }

    /// Creates an exec pool on one of the core's queues.
    ///
    /// The pool's context ring is sized from the core's worker thread count, two
    /// to eight contexts.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the pool could not be created.
    pub fn create_gpu_exec_pool(
        &self,
        queue: ffi::VSVulkanQueueType,
    ) -> Result<ExecPool, VulkanError> {
        let api = self.vulkan();
        let (handle, message) = with_error_buffer(|buffer, size| unsafe {
            (api.createGPUExecPool)(self.as_ptr(), queue as i32, buffer, size)
        });

        NonNull::new(handle)
            .map(|handle| ExecPool { handle, api })
            .ok_or_else(|| VulkanError::new(message, "could not create an exec pool"))
    }

    /// Exports a semaphore so another Vulkan device or API can wait on it.
    ///
    /// `semaphore` is a raw `VkSemaphore`, typically
    /// [`Timeline::semaphore()`]; the returned handle is a file descriptor
    /// outside Windows and must be closed by the importer when it is done.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the device cannot export semaphores or the
    /// semaphore does not belong to it.
    pub fn export_gpu_semaphore(
        &self,
        semaphore: ffi::VkSemaphore,
    ) -> Result<ffi::VSVulkanExportedSemaphore, VulkanError> {
        let mut out = MaybeUninit::<ffi::VSVulkanExportedSemaphore>::uninit();

        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (self.vulkan().exportGPUSemaphore)(
                self.as_ptr(),
                semaphore,
                out.as_mut_ptr(),
                buffer,
                size,
            )
        });

        if result != 0 {
            return Err(VulkanError::new(message, "could not export the semaphore"));
        }

        // SAFETY: the call reported success, so every field was written.
        Ok(unsafe { out.assume_init() })
    }
}

impl VideoFrame {
    /// Whether this frame is host resident or GPU resident.
    #[must_use]
    pub fn residency(&self) -> ffi::VSNodeResidency {
        // SAFETY: the frame is live for the duration of the call.
        unsafe { (self.api().getFrameResidency)(self.as_ptr()) }
    }

    /// The GPU plane, when this frame is GPU resident.
    ///
    /// The buffer and its dimensions are reached through the ordinary frame
    /// functions; this is what a kernel or another API needs for the plane
    /// itself.
    ///
    /// Returns [`None`] when the frame is not GPU resident, or when the plane
    /// does not exist.
    #[must_use]
    pub fn gpu_plane(&self, plane: i32) -> Option<ffi::VSVulkanPlaneInfo> {
        let mut info = MaybeUninit::<ffi::VSVulkanPlaneInfo>::uninit();

        // SAFETY: the frame is live and the plane index is validated by the
        // library, which reports failure through the return value.
        let result =
            unsafe { (self.api().vulkan().getGPUPlane)(self.as_ptr(), plane, info.as_mut_ptr()) };

        (result == 0).then(|| unsafe { info.assume_init() })
    }

    /// Publishes the `(timeline, value)` pair that makes this plane's contents
    /// visible to whoever waits on it.
    ///
    /// This is how a plane produced outside the core's exec pool is handed over,
    /// and it is the same pair [`ExecContext::writes_plane()`] publishes on
    /// submit.
    ///
    /// # Safety
    ///
    /// The value must already be signalled, or enqueued to be signalled, on the
    /// device, with nothing the host does afterwards able to hold it up; the
    /// ordering is yours to keep. Publishing on a plane a foreign API holds is
    /// fatal.
    pub unsafe fn set_gpu_plane_producer(
        &self,
        plane: i32,
        timeline: Option<TimelineRef>,
        value: u64,
    ) {
        let timeline = timeline.map_or(null_mut(), TimelineRef::as_ptr);

        // SAFETY: the frame is live, and the pair is the caller's to guarantee.
        unsafe { (self.api().vulkan().setGPUPlaneProducer)(self.as_ptr(), plane, timeline, value) };
    }

    /// Exports a plane so another Vulkan device or API can read it.
    ///
    /// The plane stays the foreign side's from here: it may not be recorded
    /// against, returned or cached until the foreign side's completion is
    /// published with [`VideoFrame::set_gpu_plane_producer()`] or its work is
    /// finished on the host. Declaring it in an exec context is fatal, as is
    /// returning or caching the frame while holding an exec context.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the frame is not GPU resident, the plane does
    /// not exist, or the device cannot export memory.
    pub fn export_gpu_plane(&self, plane: i32) -> Result<ffi::VSVulkanExportedMemory, VulkanError> {
        let mut out = MaybeUninit::<ffi::VSVulkanExportedMemory>::uninit();

        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (self.api().vulkan().exportGPUPlane)(
                self.as_ptr(),
                plane,
                out.as_mut_ptr(),
                buffer,
                size,
            )
        });

        if result != 0 {
            return Err(VulkanError::new(message, "could not export the plane"));
        }

        // SAFETY: the call reported success, so every field was written.
        Ok(unsafe { out.assume_init() })
    }

    /// Waits on the host for this frame's GPU work to complete.
    ///
    /// This is what a filter reading results back on the host wants after
    /// submitting: it waits the producer pairs of the frame's planes, and it
    /// reports a device reset as a reset rather than as completion.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the frame is not GPU resident or the wait could
    /// not be established.
    pub fn wait_gpu_frame(&self) -> Result<(), VulkanError> {
        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (self.api().vulkan().waitGPUFrame)(self.as_ptr(), buffer, size)
        });

        if result != 0 {
            return Err(VulkanError::new(message, "could not wait for the frame"));
        }

        Ok(())
    }
}

impl VideoNode {
    /// Takes one of this node's GPU frames as an input, for reading only.
    ///
    /// This is the counterpart of [`VideoFrame::export_gpu_plane()`] on the
    /// consuming side: the frame is handed over to the foreign API and never
    /// comes back, and any other plane of it fails. It consumes one of the
    /// caller's `request_frame_filter()` requests, so it must stand in for the
    /// usual [`VideoNode::get_frame_filter()`](Node::get_frame_filter) call
    /// rather than follow it.
    ///
    /// # Safety
    ///
    /// Must not be called while holding an exec context, which is fatal. The
    /// returned frame may not be recorded against, returned or cached until the
    /// foreign side's completion is published or its work is finished on the
    /// host.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] on CPU frames and on devices without export
    /// support.
    pub unsafe fn get_exportable_frame_filter(
        &self,
        n: i32,
        ctx: &mut FrameContext,
    ) -> Result<Option<VideoFrame>, VulkanError> {
        let api = self.api().vulkan();

        let (ptr, message) = with_error_buffer(|buffer, size| unsafe {
            (api.getExportableFrameFilter)(n, self.as_ptr(), ctx.as_ptr(), buffer, size)
        });

        if !ptr.is_null() {
            // SAFETY: the call returned a frame owned by the caller.
            return Ok(Some(unsafe { VideoFrame::from_ptr(ptr, self.api()) }));
        }

        // A null frame with no message is a request that was never made, which
        // the C API documents as an ordinary outcome rather than a failure.
        match message {
            Some(message) => Err(VulkanError::new(Some(message), "no exportable frame")),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "link-vs")]
mod tests {
    use const_str::cstr;
    use testresult::TestResult;

    use super::*;

    /// A minimal compute shader. What it computes is irrelevant; the point is
    /// that the embedded glslang accepts it and the words come back.
    const KERNEL: &CStr = cstr!(
        "#version 460\n\
        layout(local_size_x = 64) in;\n\
        void main() {}\n"
    );

    /// The kernel the pipeline test builds from: the two storage buffers and the
    /// push constant a dispatch binds, written to be a copy so that the one path
    /// through it is the whole path a filter takes.
    const COPY_KERNEL: &CStr = cstr!(
        "#version 460\n\
        layout(local_size_x = 64) in;\n\
        layout(std430, set = 0, binding = 0) readonly buffer Src { uint words[]; } src;\n\
        layout(std430, set = 0, binding = 1) writeonly buffer Dst { uint words[]; } dst;\n\
        layout(push_constant) uniform Params { uint words; } p;\n\
        void main() {\n\
            uint i = gl_GlobalInvocationID.x;\n\
            if (i >= p.words) { return; }\n\
            dst.words[i] = src.words[i];\n\
        }\n"
    );

    /// A core pinned to a usable GPU device, or [`None`] on a machine with no
    /// Vulkan device at all. The tests that need one skip rather than fail,
    /// since only the enumeration below is meaningful without a device.
    fn gpu_core() -> Option<Core> {
        let api = Api::default();
        let devices = api.vulkan().enumerate_devices().ok()?;
        let Some(index) = devices.iter().position(|device| device.usable) else {
            eprintln!("no usable Vulkan device, skipping the GPU test");
            return None;
        };
        let selected = i32::try_from(index).ok()?;

        let core = Core::builder().api(api).build();
        core.set_vulkan_device(selected).ok()?;
        eprintln!("using {}", devices[index].name);

        Some(core)
    }

    #[test]
    fn enumerate_devices() -> TestResult {
        let devices = Api::default().vulkan().enumerate_devices()?;

        assert!(!devices.is_empty(), "the loader always reports something");
        for device in &devices {
            assert!(!device.name.is_empty(), "{device:?} has no name");
            if !device.usable {
                assert!(
                    device.unusable_reason.is_some(),
                    "{device:?} is unusable with no reason"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn core_info_and_handles() -> TestResult {
        let Some(core) = gpu_core() else {
            return Ok(());
        };

        let info = core.vulkan_core_info()?;
        assert_ne!(info.device_name, "", "the device has no name");
        assert!(info.device_memory != 0);
        assert!(info.limit >= 0);

        let handles = core.vulkan_handles()?;
        assert!(!handles.device.is_null());
        assert!(!handles.physical_device.is_null());
        assert!(handles.get_instance_proc_addr.is_some());

        // Resolving these needs no loader of the caller's own.
        assert!(core.vulkan_functions().is_ok());

        Ok(())
    }

    #[test]
    fn frames_and_pool_resources() -> TestResult {
        let Some(mut core) = gpu_core() else {
            return Ok(());
        };

        assert!(core.set_max_vram_use(1 << 30) >= 0);

        let format = core.query_video_format(
            ffi::VSColorFamily::Gray,
            ffi::VSSampleType::Integer,
            8,
            0,
            0,
        );
        let frame = core
            .new_gpu_video_frame(&format, 320, 240, None)
            .expect("the core was pinned to a GPU device");

        assert_eq!(frame.residency(), ffi::VSNodeResidency::GPU);
        assert_eq!(frame.frame_width(0), 320);
        assert_eq!(frame.frame_height(0), 240);

        let plane = frame.gpu_plane(0).expect("a GPU frame has plane 0");
        assert_ne!(plane.buffer, 0);
        assert!(plane.buffer_size >= u64::try_from(frame.stride(0))? * 240);
        assert!(frame.gpu_plane(3).is_none(), "gray has one plane");

        Ok(())
    }

    #[test]
    fn buffers_shaders_timelines_and_pool() -> TestResult {
        let Some(core) = gpu_core() else {
            return Ok(());
        };

        let mut buffer = core.create_gpu_buffer(
            4096,
            ffi::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ffi::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            0,
        )?;
        assert_eq!(buffer.size(), 4096);
        assert_ne!(buffer.buffer(), 0);

        // The pool serves host visible requests mapped, which is what makes
        // uploading a table or reading a result back a plain slice write.
        let mapping = unsafe { buffer.mapped_slice() }.expect("host visible memory is mapped");
        assert_eq!(mapping.len(), 4096);
        mapping[0] = 0xAB;
        assert_eq!(mapping[0], 0xAB);

        let shader = core.compile_gpu_shader(ffi::VSGPUShaderLanguage::GLSL, KERNEL)?;
        assert_eq!(shader.code().first(), Some(&0x0723_0203), "SPIR-V magic");

        let timeline = core.create_gpu_timeline()?;
        assert_ne!(timeline.semaphore(), 0);
        assert_eq!(timeline.clone().semaphore(), timeline.semaphore());

        // Only a device that reports an exportable semaphore type can export.
        let info = core.vulkan_core_info()?;
        if info.semaphore_export_handle_type != 0 {
            let exported = core.export_gpu_semaphore(timeline.semaphore())?;
            assert_ne!(exported.handle_type, 0);
        }

        let mut reservation = core.reserve_gpu_memory(1024)?;
        reservation.update(2048);
        drop(reservation);

        let pool = core.create_gpu_exec_pool(ffi::VSVulkanQueueType::Compute)?;
        assert_eq!(pool.timeline().as_ptr(), pool.timeline().as_ptr());

        // SAFETY: this thread holds no exec context.
        let context = unsafe { pool.acquire()? };
        assert!(!context.command_buffer().is_null());
        context.abandon();

        // SAFETY: still no context held, and nothing was submitted.
        assert_ne!(
            unsafe { pool.wait_idle() }?,
            ffi::VSGPUDrainResult::DeviceLost
        );

        Ok(())
    }

    #[test]
    fn compute_pipeline_and_recording() -> TestResult {
        let Some(core) = gpu_core() else {
            return Ok(());
        };

        let format = core.query_video_format(
            ffi::VSColorFamily::Gray,
            ffi::VSSampleType::Integer,
            8,
            0,
            0,
        );
        let src = core
            .new_gpu_video_frame(&format, 256, 256, None)
            .expect("the core was pinned to a GPU device");
        let mut dst = core
            .new_gpu_video_frame(&format, 256, 256, None)
            .expect("the core was pinned to a GPU device");

        let shader = core.compile_gpu_shader(ffi::VSGPUShaderLanguage::GLSL, COPY_KERNEL)?;
        let bytes = u32::try_from(size_of::<u32>())?;
        let pipeline = ComputePipeline::builder(&core, &shader)
            .storage_buffer()
            .storage_buffer()
            .push_constants(bytes)
            .build()?;
        assert_eq!(pipeline.bindings(), 2);
        assert_ne!(pipeline.as_ptr(), 0);

        let pool = core.create_gpu_exec_pool(ffi::VSVulkanQueueType::Compute)?;
        let value = pool.record(|exec| {
            let src_plane = exec.read_plane(&src, 0)?;
            let dst_plane = exec.write_plane(&mut dst, 0)?;
            let bytes = src_plane.size().min(dst_plane.size()) / 4;
            let words = u32::try_from(bytes).expect("the planes are 256 by 256");

            exec.dispatch(
                &pipeline,
                &[src_plane, dst_plane],
                &words.to_ne_bytes(),
                [words.div_ceil(64), 1, 1],
            )
        })?;
        assert_ne!(value, 0);

        // A second recording on a thread that still holds one is the case
        // `acquire` is `unsafe` for, and the closure scoped call refuses it
        // instead of letting the core find it fatal.
        pool.record(|_| {
            assert!(pool.record(|_| Ok(())).is_err());
            Ok(())
        })?;

        // The write was declared, so the pool's value is the plane's producer
        // pair and waiting for the frame is waiting for the dispatch.
        dst.wait_gpu_frame()?;

        Ok(())
    }
}
