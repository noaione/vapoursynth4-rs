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
gpu = core.std.GPUUpload(clip)

# Output 0: the other answer from the same function. The argument is
# `clip:vnode:all;`, so a GPU clip stays GPU resident and nothing is inserted.
core.invert.ResidencyEcho(gpu, gpu=1).set_output(0)

# Outputs 1 and 2: the two pairings that disagree, and that the core ends the
# process over. Only a child process can read them; see the sample's tests.
core.invert.ResidencyEcho(gpu, gpu=0).set_output(1)
core.invert.ResidencyEcho(clip, gpu=1).set_output(2)
