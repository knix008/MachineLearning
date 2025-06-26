#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // 1. 얼굴 감지를 위한 Haar Cascade 분류기 로드
    cv::CascadeClassifier face_cascade;
    std::string cascade_path = "haarcascade_frontalface_alt.xml"; // XML 파일 경로

    if (!face_cascade.load(cascade_path)) {
        std::cerr << "오류: Cascade Classifier XML 파일을 로드할 수 없습니다." << std::endl;
        return -1;
    }

    // 2. 비디오 소스 열기 (웹캠 또는 동영상 파일)
    // 웹캠을 사용하려면 인자로 0을 전달합니다.
    // cv::VideoCapture cap(0);

    // 동영상 파일을 사용하려면 파일 경로를 전달합니다.
    std::string video_path = "8760577-hd_1920_1080_30fps.mp4"; // 동영상 파일 경로를 입력하세요.
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "오류: 비디오를 열 수 없습니다." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        // 3. 비디오에서 프레임 읽기
        cap >> frame;

        // 프레임이 비어있으면(동영상 끝) 루프 종료
        if (frame.empty()) {
            break;
        }

        // 4. 얼굴 감지를 위한 전처리 (그레이스케일 변환)
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray_frame, gray_frame); // 히스토그램 평활화로 대비 향상

        // 5. 얼굴 감지 수행
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray_frame, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        // 6. 감지된 얼굴 주위에 사각형 그리기
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2); // 초록색 사각형
        }

        // 7. 결과 프레임 출력
        cv::imshow("Face Detection", frame);

        // 'ESC' 키를 누르면 루프 종료
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // 자원 해제
    cap.release();
    cv::destroyAllWindows();

    return 0;
}