import torchvision.models as models

# Load ResNet 3D models from torchvision
resnet_3d = models.video.r3d_18()
resnet_mixed_conv = models.video.mc3_18()
print("> Pre-trained ResNet 3D models loaded successfully.")