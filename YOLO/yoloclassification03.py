from ultralytics import YOLO

def training():
    # Load a model
    model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="mnist160", epochs=100, imgsz=64)
    #print(results)

def validating():
    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load an official model
    model = YOLO("runs/classify/train/weights/best.pt")  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.top1  # top1 accuracy
    metrics.top5  # top5 accuracy

def predict():
    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load an official model
    model = YOLO("runs/classify/train/weights/best.pt")  # load a custom model

    # Predict with the model
    results = model("https://ultralytics.com/images/bus.jpg", save=True, imgsz=64, conf=0.5)  # predict on an image

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        print(boxes, masks, keypoints, probs, obb)
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk

if __name__ == "__main__" : 
    #training()
    #validating()
    predict()