//! A GPU filter: an invert that runs as one compute dispatch on the core's own
//! Vulkan device.
//!
//! It is the whole GPU filter shape in one place. A shader is compiled once and
//! becomes a compute pipeline; an exec pool owned for the filter's lifetime
//! hands out one recording per frame; the recording declares what it reads and
//! what it writes; and the submit's timeline value becomes the output plane's
//! producer pair, which is how every later consumer knows when the plane is
//! ready without waiting for anything.
//!
//! `std.GPUUpload` and `std.GPUDownload` bracket it in a script, because a graph
//! has to be all CPU or all GPU.

use std::{
    ffi::{CStr, c_void},
    ptr::null,
};

use vapoursynth4_rs::{
    core::CoreRef,
    ffi,
    frame::{FrameContext, VideoFrame},
    key,
    map::MapRef,
    node::{
        ActivationReason, Dependencies, Filter, FilterDependency, Node, RequestPattern, VideoNode,
    },
    vulkan::{ExecPool, Shader},
};

/// The kernel, in the dialect the core pins: `#version 460`, compiled for the
/// Vulkan 1.4 client.
///
/// It works one 32 bit word, four pixels, at a time. That is enough to be
/// correct without looking at strides at all: inverting every byte of the buffer
/// inverts every visible pixel, and the row padding that comes with it is never
/// read by anyone.
const KERNEL: &CStr = c"#version 460
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) readonly buffer Src { uint words[]; } src;
layout(std430, set = 0, binding = 1) writeonly buffer Dst { uint words[]; } dst;
layout(push_constant) uniform Params { uint words; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.words) { return; }
    dst.words[i] = ~src.words[i];
}";

/// Invocations per workgroup, which is the kernel's `local_size_x`.
const WORKGROUP: u32 = 64;

/// The words both planes are dispatched over, capped at what a single
/// `vkCmdDispatch` can express.
fn words_to_dispatch(src_bytes: u64, dst_bytes: u64) -> u32 {
    let words = src_bytes.min(dst_bytes) / u64::try_from(size_of::<u32>()).unwrap_or(4);
    u32::try_from(words).unwrap_or(u32::MAX)
}

/// Everything the filter needs once, created in `create` and destroyed in
/// `Drop`.
///
/// `VSVulkanCoreHandles` and the pipeline are raw handles, so the struct is
/// neither `Send` nor `Sync` on its own; the core owns the device, the handles
/// are immutable for its lifetime, and everything done with them is a Vulkan
/// call, which is thread safe.
struct Vulkan {
    vk: ffi::VSVulkanFunctions,
    device: ffi::VkDevice,
    shader_module: ffi::VkShaderModule,
    set_layout: ffi::VkDescriptorSetLayout,
    layout: ffi::VkPipelineLayout,
    pipeline: ffi::VkPipeline,
}

// SAFETY: see above.
unsafe impl Send for Vulkan {}
// SAFETY: see above.
unsafe impl Sync for Vulkan {}

impl Vulkan {
    /// Builds the pipeline. Every step assigns into `self` as it goes, so a
    /// failure part way through destroys exactly what was created.
    fn new(
        vk: ffi::VSVulkanFunctions,
        device: ffi::VkDevice,
        code: &[u32],
    ) -> Result<Self, &'static CStr> {
        let mut this = Self {
            vk,
            device,
            shader_module: 0,
            set_layout: 0,
            layout: 0,
            pipeline: 0,
        };

        // SAFETY: every call below is Vulkan on the core's device, given the
        // create-info structs the specification asks for. Nothing here outlives
        // the filter, and `Drop` gives each object back.
        unsafe {
            let create_shader_module = this
                .vk
                .vkCreateShaderModule
                .ok_or(c"the core did not resolve vkCreateShaderModule")?;
            let info = ffi::VkShaderModuleCreateInfo {
                sType: ffi::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext: null(),
                flags: 0,
                codeSize: size_of_val(code),
                pCode: code.as_ptr(),
            };
            if create_shader_module(device, &info, null(), &raw mut this.shader_module)
                != ffi::VK_SUCCESS
            {
                return Err(c"vkCreateShaderModule failed");
            }

            let create_set_layout = this
                .vk
                .vkCreateDescriptorSetLayout
                .ok_or(c"the core did not resolve vkCreateDescriptorSetLayout")?;
            let bindings = [
                ffi::VkDescriptorSetLayoutBinding {
                    binding: 0,
                    descriptorType: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount: 1,
                    stageFlags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                    pImmutableSamplers: null(),
                },
                ffi::VkDescriptorSetLayoutBinding {
                    binding: 1,
                    descriptorType: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount: 1,
                    stageFlags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                    pImmutableSamplers: null(),
                },
            ];
            // Push descriptors, so the two buffers are bound on the command
            // buffer itself and the filter owns no descriptor pool.
            let set_info = ffi::VkDescriptorSetLayoutCreateInfo {
                sType: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext: null(),
                flags: ffi::VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
                bindingCount: u32::try_from(bindings.len()).unwrap_or(2),
                pBindings: bindings.as_ptr(),
            };
            if create_set_layout(device, &set_info, null(), &raw mut this.set_layout)
                != ffi::VK_SUCCESS
            {
                return Err(c"vkCreateDescriptorSetLayout failed");
            }

            let create_layout = this
                .vk
                .vkCreatePipelineLayout
                .ok_or(c"the core did not resolve vkCreatePipelineLayout")?;
            let set_layouts = [this.set_layout];
            let ranges = [ffi::VkPushConstantRange {
                stageFlags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                offset: 0,
                size: u32::try_from(size_of::<u32>()).unwrap_or(4),
            }];
            let layout_info = ffi::VkPipelineLayoutCreateInfo {
                sType: ffi::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: null(),
                flags: 0,
                setLayoutCount: 1,
                pSetLayouts: set_layouts.as_ptr(),
                pushConstantRangeCount: 1,
                pPushConstantRanges: ranges.as_ptr(),
            };
            if create_layout(device, &layout_info, null(), &raw mut this.layout) != ffi::VK_SUCCESS
            {
                return Err(c"vkCreatePipelineLayout failed");
            }

            let create_pipelines = this
                .vk
                .vkCreateComputePipelines
                .ok_or(c"the core did not resolve vkCreateComputePipelines")?;
            let stage = ffi::VkPipelineShaderStageCreateInfo {
                sType: ffi::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: null(),
                flags: 0,
                stage: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                module: this.shader_module,
                pName: c"main".as_ptr(),
                pSpecializationInfo: null(),
            };
            let pipeline_info = ffi::VkComputePipelineCreateInfo {
                sType: ffi::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext: null(),
                flags: 0,
                stage,
                layout: this.layout,
                basePipelineHandle: 0,
                basePipelineIndex: -1,
            };
            // No pipeline cache: this is one pipeline, created once.
            if create_pipelines(device, 0, 1, &pipeline_info, null(), &raw mut this.pipeline)
                != ffi::VK_SUCCESS
            {
                return Err(c"vkCreateComputePipelines failed");
            }
        }

        Ok(this)
    }

    /// Records the dispatch into a command buffer the exec pool handed out.
    ///
    /// # Safety
    ///
    /// `cb` must be the command buffer of a live recording on this device, and
    /// both buffers must stay alive and unmodified until that recording's
    /// submission has completed. The caller declares exactly that with
    /// `reads_frame`, `writes_plane` and `submit`.
    unsafe fn record(
        &self,
        cb: ffi::VkCommandBuffer,
        layout: ffi::VkPipelineLayout,
        src: &ffi::VSVulkanPlaneInfo,
        dst: &ffi::VSVulkanPlaneInfo,
        words: u32,
    ) -> Result<(), &'static CStr> {
        let (bind_pipeline, push_descriptors, push_constants, dispatch) = (
            self.vk
                .vkCmdBindPipeline
                .ok_or(c"the core did not resolve vkCmdBindPipeline")?,
            self.vk
                .vkCmdPushDescriptorSet
                .ok_or(c"the core did not resolve vkCmdPushDescriptorSet")?,
            self.vk
                .vkCmdPushConstants2
                .ok_or(c"the core did not resolve vkCmdPushConstants2")?,
            self.vk
                .vkCmdDispatch
                .ok_or(c"the core did not resolve vkCmdDispatch")?,
        );

        let src_info = ffi::VkDescriptorBufferInfo {
            buffer: src.buffer,
            offset: 0,
            range: src.buffer_size,
        };
        let dst_info = ffi::VkDescriptorBufferInfo {
            buffer: dst.buffer,
            offset: 0,
            range: dst.buffer_size,
        };
        let writes = [
            ffi::VkWriteDescriptorSet {
                sType: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: null(),
                // Push descriptors have no set to write into; the field is
                // ignored, and the binding lands on the layout's set 0.
                dstSet: 0,
                dstBinding: 0,
                dstArrayElement: 0,
                descriptorCount: 1,
                descriptorType: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo: null(),
                pBufferInfo: &raw const src_info,
                pTexelBufferView: null(),
            },
            ffi::VkWriteDescriptorSet {
                sType: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: null(),
                dstSet: 0,
                dstBinding: 1,
                dstArrayElement: 0,
                descriptorCount: 1,
                descriptorType: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo: null(),
                pBufferInfo: &raw const dst_info,
                pTexelBufferView: null(),
            },
        ];

        let params = [words];
        let constants = ffi::VkPushConstantsInfo {
            sType: ffi::VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
            pNext: null(),
            layout,
            stageFlags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: u32::try_from(size_of::<u32>()).unwrap_or(4),
            pValues: params.as_ptr().cast(),
        };

        // SAFETY: the recording is live and owns nothing here; the two buffers
        // are declared on the context by the caller.
        unsafe {
            bind_pipeline(cb, ffi::VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
            push_constants(cb, &raw const constants);
            push_descriptors(
                cb,
                ffi::VK_PIPELINE_BIND_POINT_COMPUTE,
                layout,
                0,
                u32::try_from(writes.len()).unwrap_or(2),
                writes.as_ptr(),
            );
            dispatch(cb, words.div_ceil(WORKGROUP), 1, 1);
        }

        Ok(())
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        // SAFETY: each handle was created on this device and is given back
        // exactly once, after every submission that used it has completed --
        // the filter only returns frames through the core, which keeps them
        // alive until then.
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

/// An invert that runs on the GPU.
pub(crate) struct InvertGpuFilter {
    /// The GPU resident input.
    node: VideoNode,
    vulkan: Vulkan,
    /// One recording per frame, from the core's pool. Shared by every worker
    /// thread the core runs this filter on, which is why acquiring is per
    /// thread and per frame.
    pool: ExecPool,
}

impl Filter for InvertGpuFilter {
    type Error = &'static CStr;
    type FrameType = VideoFrame;
    type FilterData = ();

    /// Ties the two halves of the GPU declaration together: the core refuses a
    /// node that is called `:gpu` but not created with
    /// [`ffi::VSFilterFlags::GPUOutput`], and the other way round.
    const FILTER_FLAGS: i32 = ffi::VSFilterFlags::GPUOutput as i32;

    const NAME: &'static CStr = c"InvertGPU";
    const ARGS: &'static CStr = c"clip:vnode:gpu;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode:gpu;";

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<(), Self::Error> {
        let Ok(node) = input.get_video_node(key!(c"clip"), 0) else {
            return Err(c"InvertGPU: failed to get clip");
        };

        if node.residency() != ffi::VSNodeResidency::GPU {
            return Err(c"InvertGPU: the input must be a GPU clip; insert std.GPUUpload");
        }

        let cloned = node.clone();
        let vi = cloned.info();
        if !vapoursynth4_rs::utils::is_constant_video_format(vi) || vi.format.bytes_per_sample != 1
        {
            return Err(c"InvertGPU: only constant 8 bit formats are supported");
        }

        let vk = *core
            .vulkan_functions()
            .map_err(|_| c"InvertGPU: no Vulkan device is available")?;
        let handles = core
            .vulkan_handles()
            .map_err(|_| c"InvertGPU: no Vulkan device is available")?;

        // Compiled once per filter instance, and cached per core by source text
        // underneath, so a second instance of this filter costs nothing here.
        let shader: Shader = core
            .compile_gpu_shader(ffi::VSGPUShaderLanguage::GLSL, KERNEL)
            .map_err(|_| c"InvertGPU: could not compile the compute shader")?;

        let vulkan = Vulkan::new(vk, handles.device, shader.code())?;
        let pool = core
            .create_gpu_exec_pool(ffi::VSVulkanQueueType::Compute)
            .map_err(|_| c"InvertGPU: could not create an exec pool")?;

        let filter = InvertGpuFilter { node, vulkan, pool };
        let deps = [FilterDependency {
            source: filter.node.as_ptr(),
            request_pattern: RequestPattern::StrictSpatial,
        }];

        core.create_video_filter(
            output,
            c"InvertGPU",
            vi,
            Box::new(filter),
            Dependencies::new(&deps).unwrap(),
        );

        Ok(())
    }

    fn get_frame(
        &self,
        n: i32,
        activation_reason: ActivationReason,
        _frame_data: *mut *mut c_void,
        mut ctx: FrameContext,
        core: CoreRef,
    ) -> Result<Option<VideoFrame>, Self::Error> {
        use ActivationReason as r;

        match activation_reason {
            r::Initial => {
                ctx.request_frame_filter(n, &self.node);
            }
            r::AllFramesReady => {
                let src = self.node.get_frame_filter(n, &mut ctx);
                let format = src.get_video_format();
                let (width, height) = (src.frame_width(0), src.frame_height(0));

                let Some(mut dst) = core.new_gpu_video_frame(format, width, height, Some(&src))
                else {
                    return Err(c"InvertGPU: could not allocate the output frame");
                };
                // One dispatch per plane, all of them into the one recording.
                // The plane handles are taken before the context is acquired:
                // nothing that reaches the core's exec registry belongs inside
                // a recording.
                let mut planes =
                    Vec::with_capacity(usize::try_from(format.num_planes).unwrap_or(1));
                for plane in 0..format.num_planes {
                    let (Some(src_plane), Some(dst_plane)) =
                        (src.gpu_plane(plane), dst.gpu_plane(plane))
                    else {
                        return Err(c"InvertGPU: a frame is not GPU resident");
                    };
                    let words = words_to_dispatch(src_plane.buffer_size, dst_plane.buffer_size);
                    planes.push((words, src_plane, dst_plane));
                }

                // SAFETY: one recording on this thread, from acquire to submit,
                // and every declaration it carries is kept -- the buffers live
                // in the frames, which are alive here and borrowed by the
                // context, and both are released only after the submission has
                // completed.
                unsafe {
                    let exec = self
                        .pool
                        .acquire()
                        .map_err(|_| c"InvertGPU: could not acquire an exec context")?;
                    let cb = exec.command_buffer();

                    // The pool waits on the input's producer pair before the
                    // first dispatch runs, and publishes the output's own pair
                    // when it signals, so nothing else has to be synchronized.
                    exec.reads_frame(&src);

                    for (plane, (words, src_plane, dst_plane)) in
                        (0..format.num_planes).zip(&planes)
                    {
                        self.vulkan
                            .record(cb, self.vulkan.layout, src_plane, dst_plane, *words)?;
                        exec.writes_plane(&mut dst, plane);
                    }

                    exec.submit()
                        .map_err(|_| c"InvertGPU: could not submit the dispatch")?;
                }

                return Ok(Some(dst));
            }
            _ => {}
        }

        Ok(None)
    }
}
