import torchvision.models as models

# Load various pre-trained models from torchvision
faster_rcnn = models.detection.fasterrcnn_resnet50_fpn()
mask_rcnn = models.detection.maskrcnn_resnet50_fpn()
keypoint_rcnn = models.detection.keypointrcnn_resnet50_fpn()
print("> Pre-trained Faster R-CNN, Mask R-CNN, and Keypoint R-CNN models loaded successfully.")