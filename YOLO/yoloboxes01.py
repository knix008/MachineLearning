from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

def boxes():
    # Run inference on an image
    results = model("https://ultralytics.com/images/bus.jpg")  # results list

    # View results
    for r in results:
        print(r.boxes)  # print the Boxes object containing the detection bounding boxes
        
if __name__ == "__main__" :
    boxes()