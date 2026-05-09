# from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# let's generate a mesh first
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
mesh = pipeline(image="demo.png")[0]
mesh.export("demo.obj")
mesh.export("demo.glb")

print(f"Mesh generated successfully and saved to demo.obj")
print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

# Texture painting requires custom_rasterizer CUDA extension
# To enable, you need to build the custom extension:
# 1. Install Visual Studio with C++ tools
# 2. Install CUDA toolkit
# 3. Build the extension from the Hunyuan3D repository
# pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
# mesh = pipeline(mesh, image="demo.png")
# mesh.export("demo.glb")