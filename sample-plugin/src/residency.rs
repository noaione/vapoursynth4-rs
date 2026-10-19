//! A filter that decides its output residency per instance.
//!
//! `clip:vnode:gpu` and a plain `clip:vnode` each settle the question in the
//! declaration, and the framework derives the flag from it. `clip:vnode:all` is
//! the third form: it says the function can produce either, so the answer is
//! not in the type any more and has to come from the instance, through
//! [`Filter::gpu_output()`]. The node is then created with the
//! [`ffi::VSFilterFlags::GPUOutput`](vapoursynth4_rs::ffi::VSFilterFlags::GPUOutput)
//! flag or without it, and the core holds every frame the filter delivers to
//! whichever answer it was given.
//!
//! This filter answers from its own `gpu:int:opt` argument, so one function
//! hands back either a GPU frame the caller downloads with `std.GPUDownload`,
//! or an ordinary host frame for a caller that never touches the GPU.
//!
//! What it hands back is the frame it was given, unchanged. That is the
//! smallest shape whose residency can be observed: an argument declared
//! `clip:vnode:all` is neither uploaded nor downloaded, so what comes out is
//! whatever went in, and the node reports the instance's answer.
//!
//! The mismatched pairings — a GPU frame from a node without the flag, or a
//! host frame from a node with it — are ended by the core's own check, not by
//! anything here. A filter that can deliver a frame it did not allocate has to
//! be sure of the answer it gives, and one that cannot be sure should check
//! [`VideoFrame::residency()`] in `get_frame` and return an error instead,
//! because the core's refusal is fatal.

use std::ffi::{CStr, c_void};

use vapoursynth4_rs::{
    core::CoreRef,
    frame::{FrameContext, VideoFrame},
    key,
    map::MapRef,
    node::{
        ActivationReason, Dependencies, Filter, FilterDependency, Node, RequestPattern, VideoNode,
    },
};

/// A pass-through that produces host resident frames or GPU resident ones,
/// whichever its `gpu` argument asked for.
pub(crate) struct ResidencyEcho {
    /// The input, at whatever residency the caller gave it.
    node: VideoNode,
    /// What `gpu` was, and therefore what this instance produces.
    gpu: bool,
}

impl Filter for ResidencyEcho {
    type Error = &'static CStr;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"ResidencyEcho";
    /// `all` on the argument keeps the core from inserting a transfer around
    /// it: the frame arrives at the residency the caller had, which is what
    /// makes the answer below the only thing that decides.
    const ARGS: &'static CStr = c"clip:vnode:all;gpu:int:opt;";
    /// `all` is the one form that does not answer for itself, so
    /// [`Self::gpu_output()`] has to.
    const RETURN_TYPE: &'static CStr = c"clip:vnode:all;";

    fn gpu_output(&self) -> bool {
        self.gpu
    }

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<(), Self::Error> {
        let Ok(node) = input.get_video_node(key!(c"clip"), 0) else {
            return Err(c"ResidencyEcho: failed to get clip");
        };

        let n = node.clone();
        let vi = n.info();
        let filter = ResidencyEcho {
            node,
            // Absent means host resident, which is the answer for a caller that
            // did not ask for anything else.
            gpu: input.get_int(key!(c"gpu"), 0).is_ok_and(|value| value != 0),
        };
        let deps = [FilterDependency {
            source: filter.node.as_ptr(),
            request_pattern: RequestPattern::StrictSpatial,
        }];

        core.create_video_filter(
            output,
            c"ResidencyEcho",
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
        _core: CoreRef,
    ) -> Result<Option<VideoFrame>, Self::Error> {
        use ActivationReason as r;

        match activation_reason {
            r::Initial => {
                ctx.request_frame_filter(n, &self.node);
            }
            r::AllFramesReady => {
                // The frame goes back as it came: its residency is the one this
                // instance said it produces, and the core refuses the pairing
                // where the two disagree.
                return Ok(Some(self.node.get_frame_filter(n, &mut ctx)));
            }
            _ => {}
        }

        Ok(None)
    }
}
