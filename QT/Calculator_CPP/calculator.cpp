#include "calculator.h"
#include <QApplication>
#include <QMessageBox>
#include <cmath>

Calculator::Calculator(QWidget *parent)
    : QWidget(parent)
    , sumInMemory(0.0)
    , sumSoFar(0.0)
    , waitingForOperand(true)
{
    setupUI();
    connectSignals();
    setWindowTitle("Qt Calculator");
    setFixedSize(300, 400);
}

Calculator::~Calculator()
{
}

void Calculator::setupUI()
{
    // Create main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Create operation label (shows current operation)
    operationLabel = new QLabel("");
    operationLabel->setAlignment(Qt::AlignRight);
    operationLabel->setStyleSheet("QLabel { color: gray; font-size: 12px; }");
    operationLabel->setMaximumHeight(20);
    
    // Create display
    display = new QLineEdit("0");
    display->setReadOnly(true);
    display->setAlignment(Qt::AlignRight);
    display->setMaxLength(15);
    
    // Set font for display
    QFont font = display->font();
    font.setPointSize(18);
    font.setBold(true);
    display->setFont(font);
    display->setStyleSheet("QLineEdit { padding: 10px; background-color: #f0f0f0; border: 2px solid #ccc; }");
    
    // Add widgets to main layout
    mainLayout->addWidget(operationLabel);
    mainLayout->addWidget(display);
    
    // Create button grid layout
    QGridLayout *buttonLayout = new QGridLayout();
    
    // Create digit buttons (0-9)
    for (int i = 0; i < 10; ++i) {
        digitButtons[i] = new QPushButton(QString::number(i));
        digitButtons[i]->setFont(QFont("Arial", 14, QFont::Bold));
        digitButtons[i]->setMinimumSize(60, 60);
        digitButtons[i]->setStyleSheet(
            "QPushButton { "
            "background-color: #ffffff; "
            "border: 2px solid #cccccc; "
            "border-radius: 5px; "
            "} "
            "QPushButton:hover { "
            "background-color: #f0f0f0; "
            "} "
            "QPushButton:pressed { "
            "background-color: #e0e0e0; "
            "}"
        );
    }
    
    // Create operator buttons
    QString operators[4] = {"+", "-", "*", "/"};
    for (int i = 0; i < 4; ++i) {
        operatorButtons[i] = new QPushButton(operators[i]);
        operatorButtons[i]->setFont(QFont("Arial", 14, QFont::Bold));
        operatorButtons[i]->setMinimumSize(60, 60);
        operatorButtons[i]->setStyleSheet(
            "QPushButton { "
            "background-color: #ff9500; "
            "color: white; "
            "border: 2px solid #ff9500; "
            "border-radius: 5px; "
            "} "
            "QPushButton:hover { "
            "background-color: #ffad33; "
            "} "
            "QPushButton:pressed { "
            "background-color: #e6851a; "
            "}"
        );
    }
    
    // Create other buttons
    equalButton = new QPushButton("=");
    clearButton = new QPushButton("C");
    clearAllButton = new QPushButton("AC");
    backspaceButton = new QPushButton("⌫");
    decimalButton = new QPushButton(".");
    
    // Style special buttons
    equalButton->setFont(QFont("Arial", 14, QFont::Bold));
    equalButton->setMinimumSize(60, 60);
    equalButton->setStyleSheet(
        "QPushButton { "
        "background-color: #ff9500; "
        "color: white; "
        "border: 2px solid #ff9500; "
        "border-radius: 5px; "
        "} "
        "QPushButton:hover { "
        "background-color: #ffad33; "
        "} "
        "QPushButton:pressed { "
        "background-color: #e6851a; "
        "}"
    );
    
    clearButton->setFont(QFont("Arial", 12, QFont::Bold));
    clearButton->setMinimumSize(60, 60);
    clearAllButton->setFont(QFont("Arial", 12, QFont::Bold));
    clearAllButton->setMinimumSize(60, 60);
    backspaceButton->setFont(QFont("Arial", 12, QFont::Bold));
    backspaceButton->setMinimumSize(60, 60);
    decimalButton->setFont(QFont("Arial", 14, QFont::Bold));
    decimalButton->setMinimumSize(60, 60);
    
    QString specialButtonStyle = 
        "QPushButton { "
        "background-color: #a6a6a6; "
        "color: black; "
        "border: 2px solid #a6a6a6; "
        "border-radius: 5px; "
        "} "
        "QPushButton:hover { "
        "background-color: #b8b8b8; "
        "} "
        "QPushButton:pressed { "
        "background-color: #949494; "
        "}";
    
    clearButton->setStyleSheet(specialButtonStyle);
    clearAllButton->setStyleSheet(specialButtonStyle);
    backspaceButton->setStyleSheet(specialButtonStyle);
    decimalButton->setStyleSheet(
        "QPushButton { "
        "background-color: #ffffff; "
        "border: 2px solid #cccccc; "
        "border-radius: 5px; "
        "} "
        "QPushButton:hover { "
        "background-color: #f0f0f0; "
        "} "
        "QPushButton:pressed { "
        "background-color: #e0e0e0; "
        "}"
    );
    
    // Arrange buttons in grid
    // Row 0: AC, C, ⌫, /
    buttonLayout->addWidget(clearAllButton, 0, 0);
    buttonLayout->addWidget(clearButton, 0, 1);
    buttonLayout->addWidget(backspaceButton, 0, 2);
    buttonLayout->addWidget(operatorButtons[3], 0, 3); // /
    
    // Row 1: 7, 8, 9, *
    buttonLayout->addWidget(digitButtons[7], 1, 0);
    buttonLayout->addWidget(digitButtons[8], 1, 1);
    buttonLayout->addWidget(digitButtons[9], 1, 2);
    buttonLayout->addWidget(operatorButtons[2], 1, 3); // *
    
    // Row 2: 4, 5, 6, -
    buttonLayout->addWidget(digitButtons[4], 2, 0);
    buttonLayout->addWidget(digitButtons[5], 2, 1);
    buttonLayout->addWidget(digitButtons[6], 2, 2);
    buttonLayout->addWidget(operatorButtons[1], 2, 3); // -
    
    // Row 3: 1, 2, 3, +
    buttonLayout->addWidget(digitButtons[1], 3, 0);
    buttonLayout->addWidget(digitButtons[2], 3, 1);
    buttonLayout->addWidget(digitButtons[3], 3, 2);
    buttonLayout->addWidget(operatorButtons[0], 3, 3); // +
    
    // Row 4: 0 (spans 2 columns), ., =
    buttonLayout->addWidget(digitButtons[0], 4, 0, 1, 2);
    buttonLayout->addWidget(decimalButton, 4, 2);
    buttonLayout->addWidget(equalButton, 4, 3);
    
    mainLayout->addLayout(buttonLayout);
}

void Calculator::connectSignals()
{
    // Connect digit buttons
    for (int i = 0; i < 10; ++i) {
        connect(digitButtons[i], &QPushButton::clicked, this, &Calculator::digitClicked);
    }
    
    // Connect operator buttons
    for (int i = 0; i < 4; ++i) {
        connect(operatorButtons[i], &QPushButton::clicked, this, &Calculator::operatorClicked);
    }
    
    // Connect other buttons
    connect(equalButton, &QPushButton::clicked, this, &Calculator::equalClicked);
    connect(clearButton, &QPushButton::clicked, this, &Calculator::clearClicked);
    connect(clearAllButton, &QPushButton::clicked, this, &Calculator::clearAllClicked);
    connect(backspaceButton, &QPushButton::clicked, this, &Calculator::backspaceClicked);
    connect(decimalButton, &QPushButton::clicked, this, &Calculator::decimalClicked);
}

void Calculator::digitClicked()
{
    QPushButton *clickedButton = qobject_cast<QPushButton *>(sender());
    int digitValue = clickedButton->text().toInt();
    
    if (display->text() == "0" && digitValue == 0.0)
        return;
    
    if (waitingForOperand) {
        display->clear();
        waitingForOperand = false;
    }
    
    display->setText(display->text() + QString::number(digitValue));
}

void Calculator::operatorClicked()
{
    QPushButton *clickedButton = qobject_cast<QPushButton *>(sender());
    QString clickedOperator = clickedButton->text();
    double operand = display->text().toDouble();
    
    if (!pendingMultiplicativeOperator.isEmpty()) {
        if (!calculate(operand, pendingMultiplicativeOperator)) {
            clearAllClicked();
            return;
        }
        display->setText(QString::number(sumSoFar));
        operand = sumSoFar;
        sumSoFar = 0.0;
        pendingMultiplicativeOperator.clear();
    }
    
    if (!pendingAdditiveOperator.isEmpty()) {
        if (!calculate(operand, pendingAdditiveOperator)) {
            clearAllClicked();
            return;
        }
        display->setText(QString::number(sumInMemory));
    } else {
        sumInMemory = operand;
    }
    
    pendingAdditiveOperator = clickedOperator;
    if (clickedOperator == "*" || clickedOperator == "/") {
        pendingMultiplicativeOperator = clickedOperator;
        pendingAdditiveOperator.clear();
        sumSoFar = operand;
    }
    
    // Update operation label
    operationLabel->setText(QString::number(sumInMemory) + " " + clickedOperator);
    
    waitingForOperand = true;
}

void Calculator::equalClicked()
{
    double operand = display->text().toDouble();
    
    if (!pendingMultiplicativeOperator.isEmpty()) {
        if (!calculate(operand, pendingMultiplicativeOperator)) {
            clearAllClicked();
            return;
        }
        operand = sumSoFar;
        sumSoFar = 0.0;
        pendingMultiplicativeOperator.clear();
    }
    
    if (!pendingAdditiveOperator.isEmpty()) {
        if (!calculate(operand, pendingAdditiveOperator)) {
            clearAllClicked();
            return;
        }
        pendingAdditiveOperator.clear();
    } else {
        sumInMemory = operand;
    }
    
    display->setText(QString::number(sumInMemory));
    operationLabel->clear();
    sumInMemory = 0.0;
    waitingForOperand = true;
}

void Calculator::clearClicked()
{
    if (waitingForOperand)
        return;
    
    display->setText("0");
    waitingForOperand = true;
}

void Calculator::clearAllClicked()
{
    sumInMemory = 0.0;
    sumSoFar = 0.0;
    pendingAdditiveOperator.clear();
    pendingMultiplicativeOperator.clear();
    display->setText("0");
    operationLabel->clear();
    waitingForOperand = true;
}

void Calculator::backspaceClicked()
{
    if (waitingForOperand)
        return;
    
    QString text = display->text();
    text.chop(1);
    if (text.isEmpty()) {
        text = "0";
        waitingForOperand = true;
    }
    display->setText(text);
}

void Calculator::decimalClicked()
{
    if (waitingForOperand)
        display->setText("0");
    
    if (!display->text().contains('.'))
        display->setText(display->text() + tr("."));
    
    waitingForOperand = false;
}

bool Calculator::calculate(double rightOperand, const QString &pendingOperator)
{
    if (pendingOperator == "+") {
        sumInMemory += rightOperand;
    } else if (pendingOperator == "-") {
        sumInMemory -= rightOperand;
    } else if (pendingOperator == "*") {
        sumSoFar *= rightOperand;
    } else if (pendingOperator == "/") {
        if (rightOperand == 0.0) {
            QMessageBox::warning(this, tr("Calculator"), tr("Division by zero error"));
            return false;
        }
        sumSoFar /= rightOperand;
    }
    
    return true;
}
