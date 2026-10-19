import platform
import vapoursynth as vs
from vapoursynth import core

if platform.system() == "Windows":
    path = r"../target/debug/sample_plugin.dll"
elif platform.system() == "Linux":
    path = r"../target/debug/libsample_plugin.so"
else:
    path = r"../target/debug/libsample_plugin.dylib"

core.std.LoadPlugin(path)

clip = core.std.BlankClip()

# The filter panics with a formatted message instead of returning an error, and
# the invoke turns that into the script's error rather than into a dead process.
core.invert.Panic(clip, level=7)
