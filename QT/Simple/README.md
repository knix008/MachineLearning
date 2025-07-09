# 간단한 QT 애플리케이션들

이 폴더에는 다양한 GUI 애플리케이션 예제들이 포함되어 있습니다.

## 파일 목록

### 1. basic_app.py
가장 기본적인 GUI 애플리케이션입니다.

**기능:**
- 사용자 정보 입력 (이름, 나이, 성별)
- 정보 저장 및 표시
- 현재 시간 확인
- 폼 초기화

**실행 방법:**
```bash
python basic_app.py
```

**필요한 라이브러리:**
- tkinter (Python 기본 라이브러리)

### 2. simple_app.py
더 고급 기능을 포함한 GUI 애플리케이션입니다.

**기능:**
- 사용자 정보 입력 (이름, 나이, 성별, 취미)
- 정보 저장 및 표시
- 현재 시간 확인
- 폼 초기화
- 스크롤 가능한 결과 창

**실행 방법:**
```bash
python simple_app.py
```

**필요한 라이브러리:**
- tkinter (Python 기본 라이브러리)

### 3. simple_calculator.py
PyQt5를 사용한 계산기 애플리케이션입니다.

**기능:**
- 기본 사칙연산 (덧셈, 뺄셈, 곱셈, 나눗셈)
- 퍼센트 계산
- 부호 변경
- 다크 테마 UI

**실행 방법:**
```bash
python simple_calculator.py
```

**필요한 라이브러리:**
- PyQt5 (설치 필요: `pip install PyQt5`)

## 설치 및 실행

### 기본 애플리케이션 (tkinter 사용)
tkinter는 Python에 기본으로 포함되어 있으므로 추가 설치가 필요하지 않습니다.

```bash
# basic_app.py 실행
python basic_app.py

# simple_app.py 실행
python simple_app.py
```

### PyQt5 계산기 애플리케이션
PyQt5가 설치되어 있지 않은 경우 다음 명령어로 설치하세요:

```bash
pip install PyQt5
```

설치 후 실행:
```bash
python simple_calculator.py
```

## 애플리케이션 특징

### basic_app.py
- 가장 간단한 구조
- 기본적인 입력 폼
- 직관적인 UI
- 에러 처리 포함

### simple_app.py
- 더 많은 입력 필드
- 체크박스를 이용한 다중 선택
- 향상된 레이아웃
- 스크롤바가 있는 결과 창

### simple_calculator.py
- 전문적인 계산기 UI
- 다크 테마
- 호버 효과
- 완전한 계산기 기능

## 사용 팁

1. **입력 검증**: 모든 애플리케이션에는 입력 검증이 포함되어 있습니다.
2. **에러 처리**: 잘못된 입력에 대한 적절한 에러 메시지가 표시됩니다.
3. **사용자 친화적**: 직관적인 인터페이스로 쉽게 사용할 수 있습니다.

## 문제 해결

### PyQt5 관련 오류
PyQt5가 설치되지 않은 경우:
```bash
pip install PyQt5
```

### tkinter 관련 오류
대부분의 Python 설치에는 tkinter가 포함되어 있습니다. 만약 문제가 있다면:
- Python을 재설치하거나
- 시스템 패키지 관리자를 통해 tkinter를 설치하세요

## 확장 가능성

이 기본 애플리케이션들을 기반으로 다음과 같은 기능을 추가할 수 있습니다:

- 데이터베이스 연동
- 파일 저장/불러오기
- 네트워크 통신
- 더 복잡한 계산 기능
- 차트 및 그래프 표시
- 다국어 지원 