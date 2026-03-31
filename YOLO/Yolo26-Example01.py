from ultralytics import YOLO
import cv2


def run_detection_on_image(image_path: str, model_name: str = "yolo26n.pt") -> None:
    model = YOLO(model_name)

    results = model.predict(source=image_path, conf=0.25, save=True)

    for result in results:
        boxes = result.boxes
        print(f"탐지된 객체 수: {len(boxes)}")

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  - {cls_name} (신뢰도: {conf:.2f}) | 위치: [{x1}, {y1}, {x2}, {y2}]")


def run_detection_on_webcam(model_name: str = "yolo26n.pt") -> None:
    model = YOLO(model_name)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("웹캠 탐지 시작 (종료: 'q')")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 이미지 파일로 탐지
    # run_detection_on_image("test.jpg")

    # 웹캠으로 실시간 탐지
    run_detection_on_webcam()