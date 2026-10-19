use std::{
    ffi::{CStr, c_void},
    ptr::null_mut,
};

use crate::{
    core::CoreRef,
    ffi,
    frame::{Frame, FrameContext},
    map::MapRef,
    node::FilterMode,
};

pub trait Filter
where
    Self: Sized + std::panic::RefUnwindSafe,
{
    const FILTER_MODE: FilterMode = FilterMode::Parallel;

    /// Filter error that can turned into a [`&CStr`](std::ffi::CStr)
    type Error: AsRef<CStr>;
    type FrameType: Frame;
    type FilterData; // TODO: Ensure Send + Sync when FILTER_MODE is Parallel

    const NAME: &'static CStr;
    const ARGS: &'static CStr;
    /// The type of the value this filter returns, in the syntax
    /// `registerFunction` takes, such as `clip:vnode` for a video node.
    ///
    /// A `gpu` modifier on a returned clip, as in `clip:vnode:gpu`, is what
    /// makes the filter a GPU filter: the framework reads it and creates the
    /// node with [`ffi::VSFilterFlags::GPUOutput`], which is the flag the core
    /// requires the declaration to agree with. Stating it once, here, is what
    /// keeps a GPU filter from saying one thing and creating the other.
    const RETURN_TYPE: &'static CStr;

    /// # Errors
    ///
    /// Return [`Self::Error`] if anything happens during the filter creation.
    /// The error message will be passed to `VapourSynth`.
    fn create(
        input: MapRef,
        output: MapRef,
        data: Option<Box<Self::FilterData>>,
        core: CoreRef,
    ) -> Result<(), Self::Error>;

    /// # Errors
    ///
    /// Return [`Self::Error`] if anything happens during the filter creation.
    /// The error message will be passed to `VapourSynth`.
    fn get_frame(
        &self,
        n: i32,
        activation_reason: ffi::VSActivationReason,
        frame_data: *mut *mut c_void,
        frame_ctx: FrameContext,
        core: CoreRef,
    ) -> Result<Option<Self::FrameType>, Self::Error>;
    /// Free the filter
    fn free(self, core: CoreRef) {
        let _ = core;
    }
}

pub struct FilterRegister<F: Filter> {
    data: Option<F::FilterData>,
}

impl<F: Filter> FilterRegister<F> {
    pub fn new(data: Option<F::FilterData>) -> Self {
        Self { data }
    }

    /// # Safety
    pub unsafe fn register(self, plugin: *mut ffi::VSPlugin, vspapi: *const ffi::VSPLUGINAPI) {
        use super::internal::FilterExtern;

        unsafe {
            ((*vspapi).registerFunction)(
                F::NAME.as_ptr(),
                F::ARGS.as_ptr(),
                F::RETURN_TYPE.as_ptr(),
                F::filter_create,
                self.data
                    .map_or(null_mut(), |d| Box::into_raw(Box::new(d)).cast()),
                plugin,
            )
        };
    }
}

/// Whether a declared return type asks for GPU resident frames.
///
/// The syntax is `name:type[:modifier]*`, separated by `;` between returned
/// values, so the modifiers are everything after the first two parts of an
/// entry.
#[cfg(feature = "vs-43")]
fn declares_gpu(return_type: &[u8]) -> bool {
    return_type
        .split(|byte| *byte == b';')
        .filter(|entry| !entry.is_empty())
        .any(|entry| {
            let mut parts = entry.split(|byte| *byte == b':');
            parts.next();
            parts.next();
            parts.any(|part| part == b"gpu")
        })
}

/// The [`ffi::VSFilterFlags`] a filter has to be created with.
///
/// `VapourSynth` takes the requirement from the declared return type and then
/// refuses a node created without the matching flag, so the two have to agree.
/// Reading the same declaration here means a filter states it in one place and
/// cannot get the pair wrong.
/// The result is the flags argument of the entry point that takes one, which
/// `vs-43` and up do. A filter with no flags is exactly what the older entry
/// point creates, so the two paths differ only in whether the argument exists.
#[cfg(feature = "vs-43")]
pub(crate) fn filter_flags<F: Filter>() -> i32 {
    if declares_gpu(F::RETURN_TYPE.to_bytes()) {
        ffi::VSFilterFlags::GPUOutput as i32
    } else {
        0
    }
}

pub type ActivationReason = ffi::VSActivationReason;

#[cfg(test)]
#[cfg(feature = "vs-43")]
mod tests {
    use super::declares_gpu;

    #[test]
    fn gpu_modifier_is_read_from_the_type() {
        assert!(declares_gpu(b"clip:vnode:gpu;"));
        assert!(declares_gpu(b"clip:vnode:gpu"));
        assert!(declares_gpu(b"clip:vnode:gpu:whatever;alpha:vnode;"));
        assert!(declares_gpu(b"alpha:vnode;clip:vnode:gpu;"));

        assert!(!declares_gpu(b"clip:vnode;"));
        assert!(!declares_gpu(b"clip:vnode"));
        assert!(!declares_gpu(b""));
        // The type and the name are not modifiers.
        assert!(!declares_gpu(b"clip:gpu;"));
        assert!(!declares_gpu(b"gpu:vnode;"));
        // A modifier that merely starts with `gpu` is a different modifier.
        assert!(!declares_gpu(b"clip:vnode:gpu_thing;"));
    }
}
