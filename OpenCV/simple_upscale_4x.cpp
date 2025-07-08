#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 이미지 읽기
    Mat image = imread("images/Lenna.png");
    if (image.empty()) {
        cout << "이미지를 읽을 수 없습니다!" << endl;
        return -1;
    }
    
    cout << "원본 크기: " << image.cols << " x " << image.rows << endl;
    
    // 4배 확대
    Mat upscaled;
    resize(image, upscaled, Size(), 4.0, 4.0, INTER_CUBIC);
    
    cout << "확대된 크기: " << upscaled.cols << " x " << upscaled.rows << endl;
    
    // 결과 저장
    imwrite("upscaled_4x.jpg", upscaled);
    cout << "이미지가 4배 확대되어 upscaled_4x.jpg에 저장되었습니다." << endl;
    
    return 0;
} 