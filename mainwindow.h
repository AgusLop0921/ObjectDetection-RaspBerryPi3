#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>
#include <QDebug>
#include <QFile>
#include <QElapsedTimer>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QCloseEvent>
#include <QFileDialog>
#include <opencv2/opencv.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void closeEvent(QCloseEvent *event);

    int tiempoProcesamiento;

private slots:
    void on_startBtn_pressed();

    void on_browseVideoBtn_pressed();

    void on_pbBrowseBtn_pressed();

    void on_pbtxtBrowseBtn_pressed();

    void on_classesBrowseBtn_pressed();

    void detectingObjects();

    void slot_detenerCamara();
private:
    Ui::MainWindow *ui;

    cv::dnn::Net tfNetwork;

    QGraphicsScene scene;
    QGraphicsPixmapItem pixmap;

    bool videoStopped;

    bool detecting;

};

#endif // MAINWINDOW_H
