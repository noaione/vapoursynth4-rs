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
    /// The residency modifier on a returned clip is what makes the filter a GPU
    /// filter, and it is also what the framework hands to the core:
    ///
    /// - `clip:vnode:gpu` produces GPU resident frames, and the node is created
    ///   with [`ffi::VSFilterFlags::GPUOutput`].
    /// - `clip:vnode` produces host resident frames, and the flag stays clear.
    /// - `clip:vnode:all` produces either, decided per instance by
    ///   [`gpu_output()`](Self::gpu_output).
    ///
    /// Stating it once, here, is what keeps a filter from saying one thing and
    /// creating the other: for the first two the declaration *is* the flag, and
    /// for `all` the flag is the instance's answer, which the core then holds
    /// every frame it delivers to.
    const RETURN_TYPE: &'static CStr;

    /// Whether this instance produces GPU resident frames.
    ///
    /// The default is the declaration, which is what keeps `clip:vnode:gpu` and
    /// [`ffi::VSFilterFlags::GPUOutput`] from ever disagreeing. A
    /// `clip:vnode:all` return has no answer of its own, so a filter that
    /// declares one has to say what this instance makes; saying nothing is a
    /// panic rather than a guess, because a wrong flag is a frame the core
    /// refuses far from the mistake.
    ///
    /// # Panics
    ///
    /// Panics if the filter declares `clip:vnode:all` and does not override
    /// this.
    fn gpu_output(&self) -> bool {
        match declared_residency(Self::RETURN_TYPE.to_bytes()) {
            Residency::Gpu => true,
            Residency::Cpu => false,
            Residency::All => panic!(
                "{}: a `clip:vnode:all` return has to state its output residency per instance",
                Self::NAME.to_string_lossy()
            ),
        }
    }

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

/// What a declared return type says about the frames a filter produces.
///
/// The syntax is `name:type[:modifier]*`, separated by `;` between returned
/// values, so the modifiers are everything after the first two parts of an
/// entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Residency {
    /// `clip:vnode;` — host resident, and the flag must not be set.
    Cpu,
    /// `clip:vnode:gpu;` — GPU resident, and the flag must be set.
    Gpu,
    /// `clip:vnode:all;` — either, decided per instance.
    All,
}

/// The residency a declared return type asks for.
///
/// Several returned values share one filter and therefore one flag, so a `gpu`
/// entry decides for all of them; `all` only applies when nothing demands GPU
/// output, because a `gpu` declaration is the stricter of the two.
fn declared_residency(return_type: &[u8]) -> Residency {
    let modifiers = return_type
        .split(|byte| *byte == b';')
        .filter(|entry| !entry.is_empty())
        .flat_map(|entry| {
            let mut parts = entry.split(|byte| *byte == b':');
            parts.next();
            parts.next();
            parts
        });
    let mut has_all = false;
    for modifier in modifiers {
        match modifier {
            b"gpu" => return Residency::Gpu,
            b"all" => has_all = true,
            _ => {}
        }
    }
    if has_all {
        Residency::All
    } else {
        Residency::Cpu
    }
}

/// The [`ffi::VSFilterFlags`] a filter has to be created with.
///
/// `VapourSynth` takes the requirement from the declared return type and then
/// refuses a node created without the matching flag, so the two have to agree.
/// For a `gpu` or absent modifier the declaration settles it on its own; for
/// `all` it is [`Filter::gpu_output()`] that answers, and asking that instance
/// here is what makes the flag it is created with the one it said it would
/// deliver.
///
/// The result is the flags argument of the entry point that takes one, which
/// `vs-43` and up do. A filter with no flags is exactly what the older entry
/// point creates, so the two paths differ only in whether the argument exists.
#[cfg(feature = "vs-43")]
pub(crate) fn filter_flags<F: Filter>(filter: &F) -> i32 {
    if filter.gpu_output() {
        ffi::VSFilterFlags::GPUOutput as i32
    } else {
        0
    }
}

pub type ActivationReason = ffi::VSActivationReason;

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, c_void};

    use super::{Filter, Residency, declared_residency};
    use crate::{
        core::CoreRef,
        ffi,
        frame::{FrameContext, VideoFrame},
        map::MapRef,
    };

    fn residency(return_type: &str) -> Residency {
        declared_residency(return_type.as_bytes())
    }

    #[test]
    fn a_residency_modifier_is_read_from_the_type() {
        assert_eq!(residency("clip:vnode;"), Residency::Cpu);
        assert_eq!(residency("clip:vnode"), Residency::Cpu);
        assert_eq!(residency(""), Residency::Cpu);

        assert_eq!(residency("clip:vnode:gpu;"), Residency::Gpu);
        assert_eq!(residency("clip:vnode:gpu"), Residency::Gpu);
        assert_eq!(
            residency("clip:vnode:gpu:whatever;alpha:vnode;"),
            Residency::Gpu
        );
        assert_eq!(residency("alpha:vnode;clip:vnode:gpu;"), Residency::Gpu);

        assert_eq!(residency("clip:vnode:all;"), Residency::All);
        assert_eq!(residency("clip:vnode:all:whatever;"), Residency::All);
        assert_eq!(residency("alpha:vnode;clip:vnode:all;"), Residency::All);

        // The type and the name are not modifiers.
        assert_eq!(residency("clip:gpu;"), Residency::Cpu);
        assert_eq!(residency("gpu:vnode;"), Residency::Cpu);
        // A modifier that merely starts with one is a different modifier.
        assert_eq!(residency("clip:vnode:gpu_thing;"), Residency::Cpu);
        assert_eq!(residency("clip:vnode:all_thing;"), Residency::Cpu);
        // `gpu` is the stricter of the two, so it settles the pair.
        assert_eq!(residency("clip:vnode:gpu:all;"), Residency::Gpu);
        assert_eq!(residency("clip:vnode:all;alpha:vnode:gpu;"), Residency::Gpu);
    }

    /// A filter that declares one return type and nothing else, so the default
    /// answer [`Filter::gpu_output()`] gives can be read off it.
    macro_rules! declared_filter {
        ($name:ident, $return_type:expr) => {
            struct $name;

            impl Filter for $name {
                type Error = &'static CStr;
                type FrameType = VideoFrame;
                type FilterData = ();

                const NAME: &'static CStr = c"Declared";
                const ARGS: &'static CStr = c"";
                const RETURN_TYPE: &'static CStr = $return_type;

                fn create(
                    _input: MapRef,
                    _output: MapRef,
                    _data: Option<Box<Self::FilterData>>,
                    _core: CoreRef,
                ) -> Result<(), Self::Error> {
                    Ok(())
                }

                fn get_frame(
                    &self,
                    _n: i32,
                    _activation_reason: ffi::VSActivationReason,
                    _frame_data: *mut *mut c_void,
                    _frame_ctx: FrameContext,
                    _core: CoreRef,
                ) -> Result<Option<VideoFrame>, Self::Error> {
                    Ok(None)
                }
            }
        };
    }

    declared_filter!(CpuReturn, c"clip:vnode;");
    declared_filter!(GpuReturn, c"clip:vnode:gpu;");
    declared_filter!(AllReturn, c"clip:vnode:all;");

    #[test]
    fn the_declaration_answers_unless_it_cannot() {
        assert!(!CpuReturn.gpu_output());
        assert!(GpuReturn.gpu_output());
    }

    #[test]
    #[should_panic(expected = "a `clip:vnode:all` return has to state its output residency")]
    fn an_all_return_has_to_answer_per_instance() {
        let _ = AllReturn.gpu_output();
    }
}
