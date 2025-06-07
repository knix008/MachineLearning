import torchvision.models as models

# Load the Inception v3 model with pretrained weights
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
print("> Initialized Inception v3 model with default weights.")