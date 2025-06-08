import torchvision.models as models
#from torchvision.models import EfficientNet_B0_Weights
#from torchvision.models import EfficientNet_B1_Weights
from torchvision.models import EfficientNet_B7_Weights

model = models.mnasnet1_0()
print("> EfficientNet model loaded successfully.")
#print(model)

# Loading EfficientNet models with pretrained weights
#efficientnet_b0 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
#efficientnet_b1 = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
# ...
efficientnet_b7 = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
print("> EfficientNet B7 model loaded successfully.")
print(efficientnet_b7)