import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLineEdit,
    QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class CalculatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_number = ""
        self.first_number = 0
        self.operation = ""
        self.new_number = True

    def initUI(self):
        self.setWindowTitle("간단한 계산기")
        self.setGeometry(300, 300, 300, 400)

        # 중앙 위젯 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 제목 라벨
        title_label = QLabel("계산기")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title_label)

        # 결과 표시창
        self.display = QLineEdit()
        self.display.setAlignment(Qt.AlignRight)
        self.display.setFont(QFont("Arial", 14))
        self.display.setReadOnly(True)
        self.display.setText("0")
        main_layout.addWidget(self.display)

        # 버튼 그리드 레이아웃
        button_layout = QGridLayout()

        # 버튼 텍스트 정의
        buttons = [
            ["C", "±", "%", "÷"],
            ["7", "8", "9", "×"],
            ["4", "5", "6", "-"],
            ["1", "2", "3", "+"],
            ["0", ".", "=", ""],
        ]

        # 버튼 생성 및 배치
        for i, row in enumerate(buttons):
            for j, text in enumerate(row):
                if text:  # 빈 문자열이 아닌 경우에만 버튼 생성
                    button = QPushButton(text)
                    button.setFont(QFont("Arial", 12))
                    button.setMinimumSize(60, 50)

                    # 버튼 스타일 설정
                    if text in ["÷", "×", "-", "+", "="]:
                        button.setStyleSheet(
                            """
                            QPushButton {
                                background-color: #FF9500;
                                color: white;
                                border: none;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #FFB74D;
                            }
                            QPushButton:pressed {
                                background-color: #E68900;
                            }
                        """
                        )
                    elif text == "C":
                        button.setStyleSheet(
                            """
                            QPushButton {
                                background-color: #A5A5A5;
                                color: black;
                                border: none;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #BDBDBD;
                            }
                            QPushButton:pressed {
                                background-color: #8E8E8E;
                            }
                        """
                        )
                    else:
                        button.setStyleSheet(
                            """
                            QPushButton {
                                background-color: #333333;
                                color: white;
                                border: none;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #4A4A4A;
                            }
                            QPushButton:pressed {
                                background-color: #2A2A2A;
                            }
                        """
                        )

                    button.clicked.connect(self.button_clicked)
                    button_layout.addWidget(button, i, j)

        main_layout.addLayout(button_layout)

    def button_clicked(self):
        button = self.sender()
        text = button.text()

        if text.isdigit() or text == ".":
            if self.new_number:
                self.display.setText(text)
                self.new_number = False
            else:
                current_text = self.display.text()
                if text == "." and "." in current_text:
                    return  # 이미 소수점이 있으면 무시
                self.display.setText(current_text + text)

        elif text in ["+", "-", "×", "÷"]:
            self.first_number = float(self.display.text())
            self.operation = text
            self.new_number = True

        elif text == "=":
            if self.operation and not self.new_number:
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

                # 결과가 정수인지 확인
                if result == int(result):
                    self.display.setText(str(int(result)))
                else:
                    self.display.setText(str(result))
                self.new_number = True

        elif text == "C":
            self.display.setText("0")
            self.current_number = ""
            self.first_number = 0
            self.operation = ""
            self.new_number = True

        elif text == "±":
            current_text = self.display.text()
            if current_text != "0":
                if current_text.startswith("-"):
                    self.display.setText(current_text[1:])
                else:
                    self.display.setText("-" + current_text)

        elif text == "%":
            current_value = float(self.display.text())
            result = current_value / 100
            if result == int(result):
                self.display.setText(str(int(result)))
            else:
                self.display.setText(str(result))


def main():
    app = QApplication(sys.argv)

    # 애플리케이션 스타일 설정
    app.setStyle("Fusion")

    # 다크 테마 스타일시트
    app.setStyleSheet(
        """
        QMainWindow {
            background-color: #1E1E1E;
            color: white;
        }
        QLineEdit {
            background-color: #2D2D2D;
            color: white;
            border: 2px solid #3D3D3D;
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
        }
        QLabel {
            color: white;
        }
    """
    )

    calculator = CalculatorApp()
    calculator.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
