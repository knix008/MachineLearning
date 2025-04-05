from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
    print(results)

def validate():
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load an official model
    model = YOLO("runs/pose/train/weights/best.pt")  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category
    
def predict():
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load an official model
    model = YOLO("runs/pose/train/weights/best.pt")  # load a custom model

    # Predict with the model
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    #results = model("zidane.jpg")
    #print(results)

    # Access the results
    for result in results:
        xy = result.keypoints.xy  # x and y coordinates
        xyn = result.keypoints.xyn  # normalized
        kpts = result.keypoints.data  # x, y, visibility (if available)
        print("Keypoints : ", xy)
        print("Keypoints normalized : ", xyn)
        print("Keypoints data : ", kpts)
        #result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
        
if __name__ == "__main__":
    #train()
    #validate()
    predict()