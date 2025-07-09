import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QLineEdit,
    QLabel,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class CalculatorApp(QMainWindow):
    """
    PySide6를 사용하여 만든 간단한 계산기 애플리케이션 클래스입니다.
    """

    def __init__(self):
        super().__init__()
        # 계산기 상태 변수 초기화
        self.current_number = ""
        self.first_number = 0
        self.operation = ""
        self.new_number = True
        self.initUI()

    def initUI(self):
        """
        사용자 인터페이스(UI)를 초기화하고 설정합니다.
        """
        self.setWindowTitle("간단한 계산기 (PySide6)")
        self.setGeometry(300, 300, 320, 450)

        # 중앙 위젯 및 메인 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 제목 라벨
        title_label = QLabel("계산기")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        main_layout.addWidget(title_label)

        # 결과 표시창
        self.display = QLineEdit()
        self.display.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.display.setFont(QFont("Arial", 24))
        self.display.setReadOnly(True)
        self.display.setText("0")
        self.display.setMinimumHeight(60)
        main_layout.addWidget(self.display)

        # 버튼 그리드 레이아웃
        button_layout = QGridLayout()

        # 버튼 텍스트 및 위치 정의
        buttons = {
            "C": (0, 0),
            "±": (0, 1),
            "%": (0, 2),
            "÷": (0, 3),
            "7": (1, 0),
            "8": (1, 1),
            "9": (1, 2),
            "×": (1, 3),
            "4": (2, 0),
            "5": (2, 1),
            "6": (2, 2),
            "-": (2, 3),
            "1": (3, 0),
            "2": (3, 1),
            "3": (3, 2),
            "+": (3, 3),
            "0": (4, 0, 1, 2),
            ".": (4, 2),
            "=": (4, 3),  # '0' 버튼은 2칸 차지
        }

        # 버튼 생성 및 배치
        for text, pos in buttons.items():
            button = QPushButton(text)
            button.setFont(QFont("Arial", 14))
            button.setMinimumSize(60, 60)

            # 버튼 스타일 설정
            if text in ["÷", "×", "-", "+", "="]:
                button.setStyleSheet(
                    """
                    QPushButton { background-color: #FF9500; color: white; border: none; border-radius: 30px; }
                    QPushButton:hover { background-color: #FFB74D; }
                    QPushButton:pressed { background-color: #E68900; }
                """
                )
            elif text in ["C", "±", "%"]:
                button.setStyleSheet(
                    """
                    QPushButton { background-color: #A5A5A5; color: black; border: none; border-radius: 30px; }
                    QPushButton:hover { background-color: #BDBDBD; }
                    QPushButton:pressed { background-color: #8E8E8E; }
                """
                )
            else:
                button.setStyleSheet(
                    """
                    QPushButton { background-color: #333333; color: white; border: none; border-radius: 30px; }
                    QPushButton:hover { background-color: #4A4A4A; }
                    QPushButton:pressed { background-color: #2A2A2A; }
                """
                )

            button.clicked.connect(self.button_clicked)

            # 위치 및 크기 정보에 따라 버튼 추가
            if len(pos) == 4:
                button_layout.addWidget(button, pos[0], pos[1], pos[2], pos[3])
            else:
                button_layout.addWidget(button, pos[0], pos[1])

        main_layout.addLayout(button_layout)

    def button_clicked(self):
        """
        계산기 버튼 클릭 이벤트를 처리합니다.
        """
        button = self.sender()
        text = button.text()

        if text.isdigit() or text == ".":
            self.handle_number_input(text)
        elif text in ["+", "-", "×", "÷"]:
            self.handle_operation(text)
        elif text == "=":
            self.calculate_result()
        elif text == "C":
            self.clear_all()
        elif text == "±":
            self.toggle_sign()
        elif text == "%":
            self.calculate_percentage()

    def handle_number_input(self, text):
        """숫자 및 소수점 입력을 처리합니다."""
        if self.new_number:
            if text == "0" and self.display.text() == "0":
                return
            self.display.setText(text)
            self.new_number = False
        else:
            current_text = self.display.text()
            if text == "." and "." in current_text:
                return
            self.display.setText(current_text + text)

    def handle_operation(self, op):
        """사칙연산자 입력을 처리합니다."""
        try:
            self.first_number = float(self.display.text())
            self.operation = op
            self.new_number = True
        except ValueError:
            self.display.setText("Error")

    def calculate_result(self):
        """'=' 버튼 클릭 시 결과를 계산합니다."""
        if self.operation and not self.new_number:
            try:
                second_number = float(self.display.text())
                if self.operation == "+":
                    result = self.first_number + second_number
                elif self.operation == "-":
                    result = self.first_number - second_number
                elif self.operation == "×":
                    result = self.first_number * second_number
                elif self.operation == "÷":
                    if second_number == 0:
                        self.display.setText("Error")
                        return
                    result = self.first_number / second_number

                if result == int(result):
                    self.display.setText(str(int(result)))
                else:
                    self.display.setText(str(round(result, 10)))

                self.new_number = True
                self.operation = ""
            except ValueError:
                self.display.setText("Error")

    def clear_all(self):
        """'C' 버튼 클릭 시 모든 상태를 초기화합니다."""
        self.display.setText("0")
        self.current_number = ""
        self.first_number = 0
        self.operation = ""
        self.new_number = True

    def toggle_sign(self):
        """'±' 버튼 클릭 시 부호를 변경합니다."""
        current_text = self.display.text()
        if current_text != "0":
            if current_text.startswith("-"):
                self.display.setText(current_text[1:])
            else:
                self.display.setText("-" + current_text)

    def calculate_percentage(self):
        """'%' 버튼 클릭 시 백분율을 계산합니다."""
        try:
            current_value = float(self.display.text())
            result = current_value / 100
            if result == int(result):
                self.display.setText(str(int(result)))
            else:
                self.display.setText(str(result))
        except ValueError:
            self.display.setText("Error")


def main():
    """
    애플리케이션을 실행하는 메인 함수입니다.
    """
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QMainWindow {
            background-color: #1E1E1E;
        }
        QLineEdit {
            background-color: #1E1E1E;
            color: white;
            border: none;
            padding: 10px;
            margin: 5px;
        }
        QLabel {
            color: white;
            margin: 10px;
        }
    """
    )

    calculator = CalculatorApp()
    calculator.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
