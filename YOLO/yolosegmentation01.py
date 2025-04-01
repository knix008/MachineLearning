from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

def training():
    # Train the model
    results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)

if __name__ == "__main__" :
    training()