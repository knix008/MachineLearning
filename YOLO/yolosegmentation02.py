from ultralytics import YOLO

def training():
    # Load a model
    model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
    # Train the model
    results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
    print(results)
    
def validation():
    # Load a model
    model = YOLO("yolo11n-seg.pt")  # load an official model
    model = YOLO("runs/segment/train/weights/best.pt")  # load a custom trained model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95(B)
    metrics.box.map50  # map50(B)
    metrics.box.map75  # map75(B)
    metrics.box.maps  # a list contains map50-95(B) of each category
    metrics.seg.map  # map50-95(M)
    metrics.seg.map50  # map50(M)
    metrics.seg.map75  # map75(M)
    metrics.seg.maps  # a list contains map50-95(M) of each category
    #print(metrics)

if __name__ == "__main__" :
    #training()
    validation()