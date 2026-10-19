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

The module covers what the library can check and leaves the rest `unsafe`,
because most of those calls are fatal when misused and none can be checked
against device state:

- Owned handles — `Timeline`, `Shader`, `GpuBuffer`, `GpuMemory`,
  `GpuMemoryReservation` and `ExecPool` — that give their resource back when
  dropped.
- Device and core information as owned data, through
  `VsVulkanApi::enumerate_devices()` and `Core::vulkan_core_info()`; GPU frames
  through `Core::new_gpu_video_frame()`, and their planes and residency through
  `VideoFrame::gpu_plane()` and `VideoFrame::residency()`.
- Everything that records or publishes GPU work stays `unsafe`: locking a
  queue, acquiring and submitting an exec context, publishing a plane's
  producer pair, and taking an input frame with
  `VideoNode::get_exportable_frame_filter()`.

Declaring a filter's output GPU resident (`VSFilterFlags::GPUOutput`) is not
part of the safe registration API yet; use the raw bindings for that.

[8]: https://docs.rs/vapoursynth4-rs/latest/vapoursynth4_rs/vulkan/

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
