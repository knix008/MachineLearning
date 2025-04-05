from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
    print(results)
    
if __name__ == "__main__":
    train()