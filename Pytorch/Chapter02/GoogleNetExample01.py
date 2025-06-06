import torchvision.models as models

#model = models.googlenet(pretrained=True) # This will generate a warning in PyTorch 1.13+
#model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
print("> Initialized GoogLeNet model with default weights.")