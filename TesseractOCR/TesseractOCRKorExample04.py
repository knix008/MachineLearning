#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>
#include <QFileDialog>
#include <QPixmap>
#include <QImage>
#include <QTextEdit>
#include <QBuffer>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

class OCRWidget : public QWidget {
    Q_OBJECT

public:
    OCRWidget(QWidget *parent = nullptr) : QWidget(parent) {
        QVBoxLayout *layout = new QVBoxLayout(this);

        imageLabel = new QLabel("No image loaded", this);
        imageLabel->setAlignment(Qt::AlignCenter);
        layout->addWidget(imageLabel);

        QPushButton *loadButton = new QPushButton("Load Image", this);
        layout->addWidget(loadButton);

        QPushButton *ocrButton = new QPushButton("Recognize Text", this);
        layout->addWidget(ocrButton);

        resultText = new QTextEdit(this);
        resultText->setReadOnly(true);
        layout->addWidget(resultText);

        connect(loadButton, &QPushButton::clicked, this, &OCRWidget::loadImage);
        connect(ocrButton, &QPushButton::clicked, this, &OCRWidget::recognizeText);
    }

private slots:
    void loadImage() {
        QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Images (*.png *.jpg *.bmp)");
        if (!fileName.isEmpty()) {
            loadedImage = QImage(fileName);
            imageLabel->setPixmap(QPixmap::fromImage(loadedImage).scaled(400, 400, Qt::KeepAspectRatio));
        }
    }

    void recognizeText() {
        if (loadedImage.isNull()) {
            resultText->setPlainText("No image loaded!");
            return;
        }

        // Convert QImage to Pix for Tesseract
        QByteArray ba;
        QBuffer buffer(&ba);
        loadedImage.save(&buffer, "PNG");
        Pix *pixs = pixReadMem((const unsigned char*)ba.data(), ba.size());

        tesseract::TessBaseAPI tess;
        tess.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
        tess.SetImage(pixs);

        char *outText = tess.GetUTF8Text();
        resultText->setPlainText(QString::fromUtf8(outText));

        delete [] outText;
        pixDestroy(&pixs);
    }

private:
    QLabel *imageLabel;
    QImage loadedImage;
    QTextEdit *resultText;
};

#include "main.moc"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    OCRWidget window;
    window.setWindowTitle("Tesseract OCR with Qt");
    window.resize(500, 600);
    window.show();
    return app.exec();
}