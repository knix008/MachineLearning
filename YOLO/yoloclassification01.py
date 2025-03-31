from ultralytics import YOLO

def classify():
    # Load a model
    model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="mnist160", epochs=100, imgsz=64)
    #print(results)

if __name__ == "__main__" : 
    classify()