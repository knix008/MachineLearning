#include <QApplication>
#include "calculator.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // Set application properties
    app.setApplicationName("Qt Calculator");
    app.setApplicationVersion("1.0");
    app.setOrganizationName("MachineLearning Project");
    
    // Create and show calculator
    Calculator calculator;
    calculator.show();
    
    return app.exec();
}
