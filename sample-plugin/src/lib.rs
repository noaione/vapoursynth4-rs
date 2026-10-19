mod dither;
mod gpu_invert;

use std::ffi::{CStr, c_void};

use dither::DitherFilter;
use gpu_invert::InvertGpuFilter;
use vapoursynth4_rs::{
    SampleType,
    core::CoreRef,
    declare_plugin,
    frame::{FrameContext, VideoFrame},
    key,
    map::MapRef,
    node::{
        ActivationReason, Dependencies, Filter, FilterDependency, Node, RequestPattern, VideoNode,
    },
};

struct DumbFilter {
    node: VideoNode,
    enabled: bool,
}

impl Filter for DumbFilter {
    type Error = &'static CStr;
    type FrameType = VideoFrame;
    type FilterData = ();

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<(), Self::Error> {
        let Ok(node) = input.get_video_node(key!(c"clip"), 0) else {
            return Err(c"Failed to get clip");
        };
        let n = node.clone();
        let vi = n.info();

        if !vapoursynth4_rs::utils::is_constant_video_format(vi)
            || vi.format.sample_type != SampleType::Integer
            || vi.format.bits_per_sample != 8
        {
            return Err(c"Invert: only constant format 8bit integer input supported");
        }

        let filter = DumbFilter {
            node,
            enabled: input
                .get_int(key!(c"enabled"), 0)
                .map(|v| v != 0)
                .unwrap_or(true),
        };

        let deps = [FilterDependency {
            source: filter.node.as_ptr(),
            request_pattern: RequestPattern::StrictSpatial,
        }];

        core.create_video_filter(
            output,
            c"Invert",
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
                if !self.enabled {
                    return Err(c"Not enabled");
                }

                let fi = src.get_video_format();
                let height = src.frame_height(0);
                let width = src.frame_width(0);

                let mut dst = core.new_video_frame(fi, width, height, Some(&src));

                for plane in 0..fi.num_planes {
                    let mut src_p = src.plane(plane);
                    let src_stride = src.stride(plane);
                    let mut dst_p = dst.plane_mut(plane);
                    let dst_stride = dst.stride(plane);

                    let h = src.frame_height(plane);
                    let w = src.frame_width(plane);

                    for _ in 0..h {
                        for x in 0..w as usize {
                            unsafe { *dst_p.wrapping_add(x) = !*src_p.wrapping_add(x) };
                        }

                        src_p = src_p.wrapping_offset(src_stride);
                        dst_p = dst_p.wrapping_offset(dst_stride);
                    }
                }
                return Ok(Some(dst));
            }
            _ => {}
        }

        Ok(None)
    }

    const NAME: &'static CStr = c"Filter";
    const ARGS: &'static CStr = c"clip:vnode;enabled:int:opt;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode;";
}

declare_plugin!(
    c"com.example.invert",
    c"invert",
    c"VapourSynth Filter Skeleton",
    (1, 0),
    vapoursynth4_rs::VAPOURSYNTH_API_VERSION,
    0,
    (DumbFilter, None),
    (DitherFilter, None),
    (InvertGpuFilter, None)
);

#[cfg(test)]
mod tests {
    use testresult::TestResult;
    use vapoursynth4_rs::{
        api::Api,
        frame::VideoFrame,
        node::{Node, VideoNode},
        sciprt::Script,
    };

    /// The plugin VapourSynth loads.
    ///
    /// `cargo test` builds the lib target as a test harness, not as the `cdylib`
    /// the script dlopens, so it has to be built first. Saying that beats the
    /// `dlopen` dump `LoadPlugin` produces. The choice mirrors the one in the
    /// `.vpy` scripts.
    fn plugin_path() -> &'static str {
        if cfg!(target_os = "windows") {
            "../target/debug/sample_plugin.dll"
        } else if cfg!(target_os = "macos") {
            "../target/debug/libsample_plugin.dylib"
        } else {
            "../target/debug/libsample_plugin.so"
        }
    }

    fn require_plugin() {
        let path = plugin_path();
        assert!(
            std::path::Path::new(path).exists(),
            "{path} is missing; run `cargo build -p sample-plugin` before `cargo test`"
        );
    }

    /// Whether this machine has a Vulkan device the core can use at all.
    ///
    /// The GPU test skips without one rather than failing: the bindings are pure
    /// Rust, so the crate builds and the other filters run anywhere, and only
    /// the GPU path needs a device.
    fn gpu_available() -> bool {
        Api::default()
            .vulkan()
            .enumerate_devices()
            .is_ok_and(|devices| devices.iter().any(|device| device.usable))
    }

    /// The tightly packed bytes of one plane, which is what a comparison
    /// wants: two frames of the same format need not share a stride.
    fn plane_bytes(frame: &VideoFrame, plane: i32) -> Vec<u8> {
        let stride = usize::try_from(frame.stride(plane)).unwrap_or(0);
        let width = usize::try_from(frame.frame_width(plane)).unwrap_or(0);
        let height = usize::try_from(frame.frame_height(plane)).unwrap_or(0);

        let mut bytes = Vec::with_capacity(width * height);
        // SAFETY: the plane is `stride * height` readable bytes, and every row
        // read is `width` bytes into it.
        unsafe {
            let base = frame.plane(plane);
            for row in 0..height {
                let row = std::slice::from_raw_parts(base.add(row * stride), width);
                bytes.extend_from_slice(row);
            }
        }

        bytes
    }

    /// A script output as a video node, at the residency the script gave it:
    /// asking for a GPU resident clip through the flag is the only way to see
    /// one from here, since the plain getter downloads it first.
    fn video_output(script: &Script, index: i32) -> Result<VideoNode, &'static str> {
        let node = script
            .get_output_with_flags(
                index,
                vapoursynth4_rs::ffi::VSScriptGetFlags::AllowGPUResident as i32,
            )
            .map_err(|_| "no such output")?;
        // The borrow ends with the function, and the node is reference counted.
        node.as_video().cloned().ok_or("expected a video node")
    }

    #[test]
    fn test_vsscript_works() -> TestResult {
        require_plugin();

        let vss = Script::default();
        vss.evaluate_file(c"test.vpy")?;
        let node = vss.get_output(0)?;
        let vn = node.as_video().ok_or("Expected video node")?;
        let _info = vn.info();

        Ok(())
    }

    /// The whole GPU path, end to end: the clip is uploaded, run through the
    /// GPU filter, downloaded, and compared byte for byte against the CPU
    /// inverse of the same clip.
    #[test]
    fn test_gpu_invert() -> TestResult {
        require_plugin();

        if !gpu_available() {
            eprintln!("no usable Vulkan device, skipping the GPU filter test");
            return Ok(());
        }

        let vss = Script::default();
        vss.evaluate_file(c"test_gpu.vpy")?;

        // The filter declares itself `clip:vnode:gpu`, so the node it produced
        // has to report GPU residency; the core refuses the other pairing.
        assert_eq!(
            video_output(&vss, 2)?.residency(),
            vapoursynth4_rs::ffi::VSNodeResidency::GPU
        );

        let reference = video_output(&vss, 0)?
            .get_frame(0)
            .map_err(|error| error.to_string_lossy().into_owned())?;
        let inverted = video_output(&vss, 1)?
            .get_frame(0)
            .map_err(|error| error.to_string_lossy().into_owned())?;

        for plane in 0..reference.get_video_format().num_planes {
            let source = plane_bytes(&reference, plane);
            let result = plane_bytes(&inverted, plane);

            // Without this the test would still pass if both frames came back
            // identical, which is what a missing dispatch looks like.
            assert_ne!(result, source, "plane {plane} was not touched at all");

            let expected: Vec<u8> = source.iter().map(|byte| !byte).collect();
            assert_eq!(
                result, expected,
                "plane {plane} of the GPU invert is not the inverse"
            );
        }

        Ok(())
    }
}
