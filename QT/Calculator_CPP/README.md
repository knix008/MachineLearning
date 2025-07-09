# Qt C++ Calculator

Qt를 사용하여 C++로 구현된 계산기 애플리케이션입니다.

## 기능

- 기본 산술 연산 (+, -, *, /)
- 소수점 연산 지원
- 클리어 기능 (C, AC)
- 백스페이스 기능
- 영 나누기 오류 처리
- 현대적인 UI 디자인

## 필요 조건

- Qt6 (Core, Widgets 모듈)
- CMake 3.16 이상
- C++17 지원 컴파일러 (Visual Studio 2019+, GCC 7+, Clang 5+)

## 빌드 방법

### Windows (Visual Studio)

1. Qt6을 설치합니다 (https://www.qt.io/download)
2. 명령 프롬프트에서 프로젝트 디렉토리로 이동합니다:
   ```cmd
   cd c:\Home\Projects\MachineLearning\QT\Calculator_CPP
   ```

3. 빌드 디렉토리를 생성하고 CMake를 실행합니다:
   ```cmd
   mkdir build
   cd build
   cmake .. -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/msvc2022_64"
   ```

4. 프로젝트를 빌드합니다:
   ```cmd
   cmake --build . --config Release
   ```

### Windows (MinGW)

1. Qt6과 MinGW를 설치합니다
2. 명령 프롬프트에서:
   ```cmd
   cd c:\Home\Projects\MachineLearning\QT\Calculator_CPP
   mkdir build
   cd build
   cmake .. -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/mingw_64"
   mingw32-make
   ```

### Qt Creator 사용

1. Qt Creator를 열고 "Open Project"를 선택합니다
2. `CMakeLists.txt` 파일을 선택합니다
3. 빌드 설정을 구성하고 빌드합니다

## 실행 방법

빌드가 완료되면 다음과 같이 실행합니다:

```cmd
cd build
.\calculator.exe
```

## 프로젝트 구조

```
Calculator_CPP/
├── CMakeLists.txt      # CMake 빌드 설정
├── main.cpp           # 메인 함수
├── calculator.h       # Calculator 클래스 헤더
├── calculator.cpp     # Calculator 클래스 구현
└── README.md         # 이 파일
```

## 사용법

- **숫자 버튼**: 0-9 숫자 입력
- **연산자 버튼**: +, -, *, / 연산
- **= 버튼**: 계산 결과 출력
- **C 버튼**: 현재 입력 클리어
- **AC 버튼**: 모든 내용 클리어 (All Clear)
- **⌫ 버튼**: 마지막 입력 문자 삭제 (백스페이스)
- **. 버튼**: 소수점 입력

## 주요 특징

- **모던 UI**: 둥근 모서리와 호버 효과가 있는 버튼
- **오류 처리**: 0으로 나누기 등의 오류 상황 처리
- **연산 표시**: 현재 수행 중인 연산을 상단에 표시
- **키보드 친화적**: 직관적인 버튼 배치

## 개발자 정보

이 프로젝트는 Qt6 C++ 학습 목적으로 개발되었습니다.

## 라이선스

이 프로젝트는 교육 목적으로 자유롭게 사용할 수 있습니다.
