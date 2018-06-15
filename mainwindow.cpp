#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QTimer>
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    tiempoProcesamiento(2000),
    detecting(false)
{
    ui->setupUi(this);

    videoStopped = true;
    ui->tabs->setCurrentIndex(0);
    ui->videoView->setScene(&scene);
    scene.addItem(&pixmap);

    connect(ui->pbDetenerCamara,SIGNAL(clicked(bool)),this,SLOT(slot_detenerCamara()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_startBtn_pressed()
{

    if(detecting == false)
    {
        std::cout << " - " << std::endl;
        QTimer::singleShot(tiempoProcesamiento,this,SLOT(detectingObjects()));
        detecting =true;
    }

    using namespace cv;
    using namespace dnn;
    using namespace std;

    // see the following for more on this parameters
    // https://www.tensorflow.org/tutorials/image_retraining
    const int inWidth = 300;
    const int inHeight = 300;
    const float meanVal = 127.5; // 255 divided by 2
    const float inScaleFactor = 1.0f / meanVal;
    const float confidenceThreshold = 0.5f;

    QMap< int, QString > classNames;
    QFile labelsFile( ui->classesFileEdit->text() );
    if( labelsFile.open( QFile::ReadOnly | QFile::Text) )
    {
        while( ! labelsFile.atEnd() )
        {
            QString line = labelsFile.readLine();
            classNames[ line.split( ',' )[ 0 ].trimmed().toInt() ] = line.split( ',' )[ 1 ].trimmed();
        }
        labelsFile.close();
    }

    QElapsedTimer timer;

    timer.start();
    tfNetwork = cv::dnn::readNetFromTensorflow( ui->pbFileEdit->text().toStdString(),
                                                ui->pbtxtFileEdit->text().toStdString() );

    VideoCapture video;
    if(ui->cameraRadio->isChecked())
        video.open(ui->cameraSpin->value());
    else
        video.open(ui->videoEdit->text().toStdString());

    videoStopped = false;

    Mat image;

    while(detecting ==true )
    {
        video >> image;
        Mat inputBlob = cv::dnn::blobFromImage( image,
                                                inScaleFactor,
                                                Size(inWidth, inHeight),
                                                Scalar(meanVal, meanVal, meanVal),
                                                true,
                                                false );
        tfNetwork.setInput( inputBlob );
        Mat result = tfNetwork.forward();
        Mat detections( result.size[ 2 ], result.size[ 3 ], CV_32F, result.ptr< float >() );

        for( int i = 0; i < detections.rows; i++ )
        {
            float confidence = detections.at< float >( i, 2 );

            if( confidence > confidenceThreshold )
            {
                using namespace cv;

                int objectClass = ( int )( detections.at< float >( i, 1 ) );

                int left = static_cast<int>(detections.at<float>(i, 3) * image.cols);
                int top = static_cast<int>(detections.at<float>(i, 4) * image.rows);
                int right = static_cast<int>(detections.at<float>(i, 5) * image.cols);
                int bottom = static_cast<int>(detections.at<float>(i, 6) * image.rows);

                rectangle( image, Point( left, top ), Point( right, bottom ), Scalar( 0, 255, 0 ) );
                String label = classNames[ objectClass ].toStdString();
                int baseLine = 0;
                Size labelSize = getTextSize( label, FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine );
                std::cout<<"Se detecto: " << label;
                top = max( top, labelSize.height );
                rectangle( image, Point( left, top - labelSize.height ),
                           Point( left + labelSize.width, top + baseLine ),
                           Scalar( 255, 255, 255 ), FILLED );
                putText( image, label, Point( left, top ),
                         FONT_HERSHEY_SIMPLEX, 0.5, Scalar( 0, 0, 0 ) );
            }
        }

        pixmap.setPixmap(QPixmap::fromImage(QImage(image.data,
                                                   image.cols,
                                                   image.rows,
                                                   image.step,
                                                   QImage::Format_RGB888).rgbSwapped()));
        ui->videoView->fitInView(&pixmap, Qt::KeepAspectRatio);

        qApp->processEvents();
    }

    video.release();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if(!videoStopped)
    {
        QMessageBox::warning(this,
                             "Warning",
                             "First make sure you stop the video/camera!");
        event->ignore();
    }
    else
    {
        event->accept();
    }
}

void MainWindow::on_browseVideoBtn_pressed()
{
    QString fname = QFileDialog::getOpenFileName(this, "Open Video", QString(), "Videos (*.avi *.mp4 *.mov)");
    if(QFile::exists(fname))
        ui->videoEdit->setText(fname);
}

void MainWindow::on_pbBrowseBtn_pressed()
{
    QString fname = QFileDialog::getOpenFileName(this, "Open Model", QString(), "PB files (*.pb)");
    if(QFile::exists(fname))
        ui->pbFileEdit->setText(fname);
}

void MainWindow::on_pbtxtBrowseBtn_pressed()
{
    QString fname = QFileDialog::getOpenFileName(this, "Open Config", QString(), "PBTXT files (*.pbtxt)");
    if(QFile::exists(fname))
        ui->pbtxtFileEdit->setText(fname);
}

void MainWindow::on_classesBrowseBtn_pressed()
{
    QString fname = QFileDialog::getOpenFileName(this, "Open Classes", QString(), "All files (*.*)");
    if(QFile::exists(fname))
        ui->classesFileEdit->setText(fname);
}
void MainWindow::detectingObjects()
{
    detecting = false;
    std::cout<<"detecting = false ";
}

void MainWindow::slot_detenerCamara()
{
    if(!videoStopped)
    {
        videoStopped = true;
    }
}
