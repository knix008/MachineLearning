#ifndef CALCULATOR_H
#define CALCULATOR_H

#include <QWidget>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QString>
#include <QFont>

class Calculator : public QWidget
{
    Q_OBJECT

public:
    Calculator(QWidget *parent = nullptr);
    ~Calculator();

private slots:
    void digitClicked();
    void operatorClicked();
    void equalClicked();
    void clearClicked();
    void clearAllClicked();
    void backspaceClicked();
    void decimalClicked();

private:
    void setupUI();
    void connectSignals();
    bool calculate(double rightOperand, const QString &pendingOperator);
    
    QLineEdit *display;
    QLabel *operationLabel;
    
    // Buttons
    QPushButton *digitButtons[10];
    QPushButton *operatorButtons[4]; // +, -, *, /
    QPushButton *equalButton;
    QPushButton *clearButton;
    QPushButton *clearAllButton;
    QPushButton *backspaceButton;
    QPushButton *decimalButton;
    
    // Calculator state
    double sumInMemory;
    double sumSoFar;
    QString pendingAdditiveOperator;
    QString pendingMultiplicativeOperator;
    bool waitingForOperand;
};

#endif // CALCULATOR_H
