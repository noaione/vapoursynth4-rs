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

clip = core.std.BlankClip(format=vs.GRAY8, width=64, height=64, length=1)

# One function asked for the host-resident answer twice: explicitly, and by
# leaving the optional argument out. The declaration is `clip:vnode:all;`, so
# what the node is created with is this instance's answer and not the type's.
core.invert.ResidencyEcho(clip, gpu=0).set_output(0)
core.invert.ResidencyEcho(clip).set_output(1)
