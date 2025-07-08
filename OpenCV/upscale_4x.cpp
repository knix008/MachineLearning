#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

/**
 * 이미지를 4배 확대하는 함수
 * @param input_path 입력 이미지 경로
 * @param output_path 출력 이미지 경로
 * @param interpolation 보간법 (기본값: INTER_CUBIC)
 * @return 성공 여부
 */
bool upscaleImage4x(const string& input_path, const string& output_path, int interpolation = INTER_CUBIC) {
    // 이미지 읽기
    Mat image = imread(input_path);
    if (image.empty()) {
        cout << "이미지를 읽을 수 없습니다: " << input_path << endl;
        return false;
    }
    
    // 원본 크기 출력
    cout << "원본 크기: " << image.cols << " x " << image.rows << endl;
    
    // 4배 확대
    Mat upscaled;
    resize(image, upscaled, Size(), 4.0, 4.0, interpolation);
    
    // 확대된 크기 출력
    cout << "확대된 크기: " << upscaled.cols << " x " << upscaled.rows << endl;
    
    // 결과 저장
    if (imwrite(output_path, upscaled)) {
        cout << "확대된 이미지가 저장되었습니다: " << output_path << endl;
        return true;
    } else {
        cout << "이미지 저장에 실패했습니다: " << output_path << endl;
        return false;
    }
}

/**
 * 다양한 보간법으로 4배 확대하여 비교하는 함수
 * @param input_path 입력 이미지 경로
 * @param output_dir 출력 디렉토리
 */
void compareUpscalingMethods(const string& input_path, const string& output_dir) {
    // 이미지 읽기
    Mat image = imread(input_path);
    if (image.empty()) {
        cout << "이미지를 읽을 수 없습니다: " << input_path << endl;
        return;
    }
    
    // 보간법들
    vector<pair<string, int>> methods = {
        {"nearest", INTER_NEAREST},
        {"bilinear", INTER_LINEAR},
        {"bicubic", INTER_CUBIC},
        {"lanczos", INTER_LANCZOS4}
    };
    
    // 각 방법으로 확대
    for (const auto& method : methods) {
        cout << method.first << " 방법으로 확대 중..." << endl;
        
        Mat upscaled;
        resize(image, upscaled, Size(), 4.0, 4.0, method.second);
        
        string output_path = output_dir + "/upscaled_4x_" + method.first + ".jpg";
        if (imwrite(output_path, upscaled)) {
            cout << "저장됨: " << output_path << endl;
        } else {
            cout << "저장 실패: " << output_path << endl;
        }
    }
}

/**
 * 엣지 강화를 적용하여 4배 확대하는 함수
 * @param input_path 입력 이미지 경로
 * @param output_path 출력 이미지 경로
 * @return 성공 여부
 */
bool upscaleWithEdgeEnhancement(const string& input_path, const string& output_path) {
    // 이미지 읽기
    Mat image = imread(input_path);
    if (image.empty()) {
        cout << "이미지를 읽을 수 없습니다: " << input_path << endl;
        return false;
    }
    
    // 1단계: Bicubic 보간법으로 4배 확대
    Mat upscaled;
    resize(image, upscaled, Size(), 4.0, 4.0, INTER_CUBIC);
    
    // 2단계: 언샤프 마스킹으로 엣지 강화
    Mat blurred, sharpened;
    GaussianBlur(upscaled, blurred, Size(0, 0), 1.0);
    addWeighted(upscaled, 1.5, blurred, -0.5, 0, sharpened);
    
    // 3단계: 노이즈 제거
    Mat denoised;
    fastNlMeansDenoisingColored(sharpened, denoised, 10, 10, 7, 21);
    
    // 결과 저장
    if (imwrite(output_path, denoised)) {
        cout << "엣지 강화된 확대 이미지가 저장되었습니다: " << output_path << endl;
        return true;
    } else {
        cout << "이미지 저장에 실패했습니다: " << output_path << endl;
        return false;
    }
}

/**
 * 이미지 정보를 출력하는 함수
 * @param image 이미지
 * @param title 제목
 */
void printImageInfo(const Mat& image, const string& title) {
    cout << "=== " << title << " ===" << endl;
    cout << "크기: " << image.cols << " x " << image.rows << endl;
    cout << "채널 수: " << image.channels() << endl;
    cout << "데이터 타입: " << image.type() << endl;
    cout << "메모리 사용량: " << image.total() * image.elemSize() / 1024 / 1024 << " MB" << endl;
    cout << endl;
}

int main() {
    cout << "=== OpenCV C++ 4배 이미지 확대 예제 ===" << endl << endl;
    
    // 테스트 이미지 경로
    string input_image = "images/Lenna.png";
    
    try {
        // 1. 기본 4배 확대
        cout << "1. 기본 4배 확대 (Bicubic 보간법)" << endl;
        if (upscaleImage4x(input_image, "upscaled_4x_basic.jpg")) {
            cout << "성공!" << endl;
        }
        cout << endl;
        
        // 2. 다양한 보간법 비교
        cout << "2. 다양한 보간법 비교" << endl;
        compareUpscalingMethods(input_image, "output");
        cout << endl;
        
        // 3. 엣지 강화 적용
        cout << "3. 엣지 강화 적용" << endl;
        if (upscaleWithEdgeEnhancement(input_image, "upscaled_4x_enhanced.jpg")) {
            cout << "성공!" << endl;
        }
        cout << endl;
        
        // 4. 이미지 정보 출력
        Mat original = imread(input_image);
        if (!original.empty()) {
            printImageInfo(original, "원본 이미지");
            
            Mat upscaled;
            resize(original, upscaled, Size(), 4.0, 4.0, INTER_CUBIC);
            printImageInfo(upscaled, "4배 확대된 이미지");
        }
        
        cout << "모든 처리가 완료되었습니다!" << endl;
        
    } catch (const exception& e) {
        cout << "오류가 발생했습니다: " << e.what() << endl;
        return -1;
    }
    
    return 0;
} 