//! A GPU filter: an invert that runs as one compute dispatch on the core's own
//! Vulkan device.
//!
//! It is the whole GPU filter shape in one place, and none of it is `unsafe`. A
//! shader is compiled once and becomes a [`ComputePipeline`]; an exec pool owned
//! for the filter's lifetime hands out one recording per frame; each plane pair is
//! declared and bound by the same call, so a recording cannot use a buffer it
//! never declared; and the submit's timeline value becomes every output plane's
//! producer pair, which is how a later consumer knows when a plane is ready
//! without waiting for anything.
//!
//! `std.GPUUpload` and `std.GPUDownload` bracket it in a script, because a graph
//! has to be all CPU or all GPU.

use std::ffi::{CStr, c_void};

use vapoursynth4_rs::{
    core::CoreRef,
    ffi,
    frame::{FrameContext, VideoFrame},
    key,
    map::MapRef,
    node::{
        ActivationReason, Dependencies, Filter, FilterDependency, Node, RequestPattern, VideoNode,
    },
    vulkan::{ComputePipeline, ExecPool},
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

/// An invert that runs on the GPU.
pub(crate) struct InvertGpuFilter {
    /// The GPU resident input.
    node: VideoNode,
    /// Everything built from the shader, created once and shared by every worker
    /// thread the core runs this filter on.
    pipeline: ComputePipeline,
    /// One recording per frame, from the core's pool. Shared by every worker
    /// thread the core runs this filter on, which is why acquiring is per
    /// thread and per frame.
    pool: ExecPool,
}

impl Filter for InvertGpuFilter {
    type Error = &'static CStr;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"InvertGPU";
    const ARGS: &'static CStr = c"clip:vnode:gpu;";
    /// `clip:vnode:gpu` is the whole GPU declaration. The modifier is what
    /// makes the core hand the filter a device, and it is also what makes the
    /// framework create the node with [`ffi::VSFilterFlags::GPUOutput`]; the
    /// two can never disagree because only one of them is written down.
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

        // Compiled once per filter instance, and cached per core by source text
        // underneath, so a second instance of this filter costs nothing here.
        let shader = core
            .compile_gpu_shader(ffi::VSGPUShaderLanguage::GLSL, KERNEL)
            .map_err(|_| c"InvertGPU: could not compile the compute shader")?;
        // Binding 0 is the plane read and binding 1 the plane written, in the
        // order the kernel declares them, which is also the order a dispatch
        // binds them in.
        let pipeline = ComputePipeline::builder(&core, &shader)
            .storage_buffer()
            .storage_buffer()
            .push_constants(u32::try_from(size_of::<u32>()).unwrap_or(4))
            .build()
            .map_err(|_| c"InvertGPU: could not build the compute pipeline")?;
        let pool = core
            .create_gpu_exec_pool(ffi::VSVulkanQueueType::Compute)
            .map_err(|_| c"InvertGPU: could not create an exec pool")?;

        let filter = InvertGpuFilter {
            node,
            pipeline,
            pool,
        };
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
                // Reading a plane and writing a plane is what declares both
                // frames on it, so the core keeps them alive until the
                // submission has completed and nothing else has to be
                // synchronized: the pool waits on the input's producer pairs
                // before the first dispatch runs, and publishes the output's own
                // pair when it signals.
                self.pool
                    .record(|exec| {
                        for plane in 0..format.num_planes {
                            let src_plane = exec.read_plane(&src, plane)?;
                            let dst_plane = exec.write_plane(&mut dst, plane)?;
                            let words = words_to_dispatch(src_plane.size(), dst_plane.size());

                            exec.dispatch(
                                &self.pipeline,
                                &[src_plane, dst_plane],
                                &words.to_ne_bytes(),
                                [words.div_ceil(WORKGROUP), 1, 1],
                            )?;
                        }

                        Ok(())
                    })
                    .map_err(|_| c"InvertGPU: could not record the dispatch")?;

                return Ok(Some(dst));
            }
            _ => {}
        }

        Ok(None)
    }
}
