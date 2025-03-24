from ultralytics import YOLO

def run_yolo():
    # Load a model
    model = YOLO("yolo11n.yaml")
    model = YOLO("yolo11n.pt")

    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
    return results

if __name__ == "__main__" :
    run_yolo()
