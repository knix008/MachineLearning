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

if __name__ == "__main__" : 
    #training()
    validating()