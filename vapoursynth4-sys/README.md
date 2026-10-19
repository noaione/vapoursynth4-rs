# `vapoursynth4-sys`

[![Crates](https://img.shields.io/crates/v/vapoursynth4-sys.svg)][1]
[![Documentation](https://docs.rs/vapoursynth4-sys/badge.svg)][2]

[1]: https://crates.io/crates/vapoursynth4-sys
[2]: https://docs.rs/vapoursynth4-sys

Raw bindings to [VapourSynth][3]. Special thanks to [Ivan Molodetskikh][4] for
their work on the original bindings.

[3]: https://github.com/vapoursynth/vapoursynth
[4]: https://github.com/YaLTeR/vapoursynth-rs

Check out [`vapoursynth4-rs`](https://crates.io/crates/vapoursynth4-rs) for a safe Rust wrapper.

## Supported Versions

All VapourSynth and VSScript API versions starting with 4.0 are supported,
up to VapourSynth API 4.3 and VSScript API 4.4 (R81).
By default this crate enables no features, i.e. VapourSynth API 4.0.
To use a specific version, enable the corresponding Cargo feature:

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

## Vulkan bindings

The `vs-vulkan` feature exposes the GPU side of API 4.3 (`VSVulkan4.h`) and
implies `vs-43`. `VSVulkan4.h` includes `<vulkan/vulkan_core.h>`, so the crate
also needs the Vulkan declarations that header reaches.

Those are generated from the Khronos header and checked in as
`src/vulkan.rs`; building does **not** need a Vulkan SDK, only generating does:

```sh
scripts/gen-vulkan-bindings.py \
    "$VULKAN_SDK/include/vulkan/vulkan_core.h" \
    "$VS_DIR/include/VSVulkan4.h" > src/vulkan.rs
cargo fmt -p vapoursynth4-sys
```

`src/vsvulkan.rs` is hand written in the style of the rest of the crate and is
not generated.

Because these are FFI structs, the generated layout is verified against the C
compiler rather than trusted. `scripts/check-vulkan-layout.sh` compiles a C and
a Rust probe over the same headers and diffs `sizeof`, `_Alignof` and
`offsetof` for every aggregate, and also fails if the checked-in file is stale
(the comparison is against the generator's output, formatted, so regenerating
and running `cargo fmt` is all it takes to satisfy it):

```sh
scripts/check-vulkan-layout.sh "$VULKAN_SDK/include" "$VS_DIR/include"
```

Run it after bumping the VapourSynth or Vulkan SDK version. A layout change
there is an ABI change, not a refactor.

## License

Licensed under [MPL-2.0](LICENSE) or at <http://mozilla.org/MPL/2.0/>.
