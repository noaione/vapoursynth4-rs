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

# Output 0 is the plain CPU clip the GPU result is compared against.
clip = core.std.BlankClip(format=vs.YUV420P8, width=320, height=240, length=3)
clip.set_output(0)

# A graph is all CPU or all GPU, so the clip is uploaded, inverted on the GPU,
# and downloaded again before anything on the host reads it.
gpu = core.std.GPUUpload(clip)
gpu = core.invert.InvertGPU(gpu)

# Output 2 is the GPU node itself, to check that it really is GPU resident.
gpu.set_output(2)

cpu = core.std.GPUDownload(gpu)
cpu.set_output(1)
