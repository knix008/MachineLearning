from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

def predict():
    # Run batched inference on a list of images
    #results = model.predict("https://ultralytics.com/images/bus.jpg", save=True, imgsz=320, conf=0.5, device="cuda:0")  # return a list of Results objects
    #results = model.predict("Test01.png", save=True, imgsz=160, conf=0.5, device="cuda:0")
    results = model.predict("Test02.jpg", save=True, imgsz=640, conf=0.5, device="cuda:0")
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk


if __name__ == "__main__" :
    predict()
