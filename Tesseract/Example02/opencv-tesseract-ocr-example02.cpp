/*
  a simple program demonstrating the use of Tesseract OCR engine with OpenCV,
  adopted from http://www.sk-spell.sk.cx/simple-example-how-to-call-use-tesseract-library
 */

#include <baseapi.h>
#include <iostream>
// #include <allheaders.h>
#include <sys/time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
  // initilize tesseract OCR engine
  tesseract::TessBaseAPI *myOCR =
      new tesseract::TessBaseAPI();

  printf("Tesseract-ocr version: %s\n", myOCR->Version());
  if (myOCR->Init(NULL, "kor+eng"))
  {
    fprintf(stderr, "Could not initialize tesseract.\n");
    exit(1);
  }

  tesseract::PageSegMode pagesegmode = static_cast<tesseract::PageSegMode>(7); // treat the image as a single text line
  myOCR->SetPageSegMode(pagesegmode);

  // read iamge
  namedWindow("tesseract-opencv", 0);
  Mat image = imread("../DriverLicense-Sample01.jpg", 0);

  // set region of interest (ROI), i.e. regions that contain text
  Rect text1ROI(350, 30, 400, 40);

  // recognize text
  myOCR->TesseractRect(image.data, 1, image.step1(), text1ROI.x, text1ROI.y, text1ROI.width, text1ROI.height);
  const char *text1 = myOCR->GetUTF8Text();

  // remove "newline"
  string t1(text1);
  t1.erase(std::remove(t1.begin(), t1.end(), '\n'), t1.end());

  // print found text
  printf("found text1: \n");
  cout << t1.c_str() << endl;
  printf("\n");

  // draw text on original image
  Mat scratch = imread("../DriverLicense-Sample01.jpg");

  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 2;
  int thickness = 2;
  putText(scratch, t1, Point(text1ROI.x, text1ROI.y), fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
  rectangle(scratch, text1ROI, Scalar(0, 0, 255), 2, 8, 0);

  imshow("tesseract-opencv", scratch);
  imwrite("result.jpg", scratch);
  waitKey(0);

  delete[] text1;

  // destroy tesseract OCR engine
  myOCR->Clear();
  myOCR->End();

  return 0;
}