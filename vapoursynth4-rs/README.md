# `vapoursynth4-rs`

[![Crates](https://img.shields.io/crates/v/vapoursynth4-rs.svg)][1]
[![Documentation](https://docs.rs/vapoursynth4-rs/badge.svg)][2]
[![dependency status](https://deps.rs/repo/github/inflation/vapoursynth4-rs/status.svg)][3]
[![CI](https://github.com/inflation/vapoursynth4-rs/workflows/CI/badge.svg)][4]
[![License: MPL-2.0](https://img.shields.io/crates/l/vapoursynth4-rs)][5]

[1]: https://crates.io/crates/vapoursynth4-rs
[2]: https://docs.rs/vapoursynth4-rs
[3]: https://deps.rs/repo/github/inflation/vapoursynth4-rs
[4]: https://github.com/inflation/vapoursynth4-rs/actions?query=workflow%3ACI
[5]: https://github.com/inflation/vapoursynth4-rs/blob/master/LICENSE

Safe wrapper to [VapourSynth][6]. Special thanks to [Ivan Molodetskikh][7] for
their work on the original bindings.

[6]: https://github.com/vapoursynth/vapoursynth
[7]: https://github.com/YaLTeR/vapoursynth-rs

Check out [`vapoursynth4-sys`](https://crates.io/crates/vapoursynth4-sys) for the raw binding.

## Supported Versions

All VapourSynth and VSScript API versions starting with 4.0 are supported,
up to VapourSynth API 4.3 and VSScript API 4.4 (R81).
By default, the crate uses the latest API version available.
To use a specific version,
disable the default features and enable the corresponding Cargo feature:

- `vs-41` for VapourSynth API 4.1 (R66)
- `vs-42` for VapourSynth API 4.2 (R74)
- `vs-43` for VapourSynth API 4.3 (R80)
- `vsscript` for VSScript API 4.1, the base version
- `vsscript-42` for VSScript API 4.2 (R69)
- `vsscript-43` for VSScript API 4.3 (R79)
- `vsscript-44` for VSScript API 4.4 (R80)
- `vs-graph` for the experimental graph inspection API (implies `vs-42`)
- `vs-vulkan` for the `VSVulkan4` GPU API (implies `vs-43`)

Each version feature enables the ones below it. Note that API 4.2 replaces the
`_ColorRange` frame property with `_Range`, whose two values are the other way
around; enabling `vs-42` deprecates `VSColorRange` in favour of `VSRange`.

## GPU filters

Enabling `vs-vulkan` adds the [`vulkan`][8] module, the safe wrapper over
`VSVulkan4.h`. It is opt-in rather than part of the default set, so nothing
changes for a filter that only ever produces CPU frames.

The module covers what the library can check, and leaves the rest `unsafe` where
nothing but the caller knows the answer:

- Owned handles — `Timeline`, `Shader`, `GpuBuffer`, `GpuMemory`,
  `GpuMemoryReservation` and `ExecPool` — that give their resource back when
  dropped, and a [`ComputePipeline`][9] that builds the shader module, both
  layouts and the pipeline from a compiled shader.
- Device and core information as owned data, through
  `VsVulkanApi::enumerate_devices()` and `Core::vulkan_core_info()`; GPU frames
  through `Core::new_gpu_video_frame()`, and their planes and residency through
  `VideoFrame::gpu_plane()` and `VideoFrame::residency()`.
- A whole recording, through `ExecPool::record()`: it claims the context for the
  calling thread, so the one context per thread rule that makes `acquire`
  `unsafe` is checked rather than documented. Inside it, `read_plane()` and
  `write_plane()` hand back the plane to bind and declaring it on the recording
  is part of handing it back, so a submission cannot bind a buffer it never
  declared, and `dispatch()` records it.
- What is left `unsafe` is what only a caller can vouch for: locking a queue,
  acquiring a context by hand and submitting it, handing over a buffer or a
  memory region, publishing a plane's producer pair, and taking an input frame
  with `VideoNode::get_exportable_frame_filter()`.

A GPU filter declares itself once: the residency modifier on its return type.
`clip:vnode:gpu` means GPU resident frames and `clip:vnode` means host ones,
and the `VSFilterFlags::GPUOutput` the core also wants is read from that
declaration, so the node and the type cannot disagree. `clip:vnode:all` is the
third form — the function can produce either — and that one is answered per
instance by [`Filter::gpu_output()`][10]: a registered function can hand back a
GPU frame for a caller that downloads it, or an ordinary host frame for a
caller that never touches the device.

The example in `sample-plugin` — `src/gpu_invert.rs`, run by `test_gpu.vpy` —
does the whole thing with no `unsafe` in it at all: a shader compiled once, a
pipeline built from it, one dispatch per plane recorded into a single
submission, and a result compared byte for byte against the CPU inverse of the
same clip. `src/residency.rs` is the smaller one, a pass-through that answers
`all` from its own argument, run by `test_residency.vpy` and
`test_residency_gpu.vpy`.

[8]: https://docs.rs/vapoursynth4-rs/latest/vapoursynth4_rs/vulkan/
[9]: https://docs.rs/vapoursynth4-rs/latest/vapoursynth4_rs/vulkan/struct.ComputePipeline.html
[10]: https://docs.rs/vapoursynth4-rs/latest/vapoursynth4_rs/node/trait.Filter.html#method.gpu_output

## Building

Linking is optional and enabled with the `link-vs` and `link-vsscript` features.
If you enable them, make sure the corresponding libraries are available.

The simplest way to get them on any platform is the official wheel, which is the
same payload the Windows installer and the portable zip ship since R74:

```sh
pip install vapoursynth==81
```

Use the `VAPOURSYNTH_LIB_PATH` environment variable to point at the directory
holding the library files.

On Windows no import library is needed, and none is shipped any more: the DLLs
are bound directly by name (`libvapoursynth.dll` and `vsscript.dll`), so it is
enough for them to be findable on `PATH` at run time.

On Linux and macOS the wheel ships only the runtime sonames
(`libvapoursynth.so.4`, `libvsscript.so`), so linking against it needs the usual
development symlinks — see `scripts/install-vs.sh` for the exact set.

## License

Licensed under [MPL-2.0](LICENSE) or at <http://mozilla.org/MPL/2.0/>.
