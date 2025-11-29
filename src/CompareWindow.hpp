#pragma once

#include <QDialog>
#include <QLabel>
#include <QTableWidget>
#include <opencv2/core.hpp>

#include "Stats.hpp"   // Usa TissueStats y SliceStats

class CompareWindow : public QDialog {
    Q_OBJECT

public:
    // parent        : ventana principal
    // origGray      : imagen original en gris
    // gaussGray     : suavizado gaussiano
    // nlGray        : suavizado NLMeans
    // dncnnGray     : suavizado DnCNN
    // overlayOrig   : overlay color original
    // overlayGauss  : overlay color gaussiano
    // overlayNL     : overlay color NLMeans
    // overlayDncnn  : overlay color DnCNN
    explicit CompareWindow(QWidget* parent,
                           const cv::Mat& origGray,
                           const cv::Mat& gaussGray,
                           const cv::Mat& nlGray,
                           const cv::Mat& dncnnGray,
                           const cv::Mat& overlayOrig,
                           const cv::Mat& overlayGauss,
                           const cv::Mat& overlayNL,
                           const cv::Mat& overlayDncnn,
                           const SliceStats& stats);

private:
    // Overlays (fila de arriba)
    QLabel* m_lblOverlayOrig   = nullptr;
    QLabel* m_lblOverlayGauss  = nullptr;
    QLabel* m_lblOverlayNL     = nullptr;
    QLabel* m_lblOverlayDncnn  = nullptr;

    // Visual grayscale (fila de abajo)
    QLabel* m_lblGrayOrig      = nullptr;
    QLabel* m_lblGrayGauss     = nullptr;
    QLabel* m_lblGrayNL        = nullptr;
    QLabel* m_lblGrayDncnn     = nullptr;

    QTableWidget* m_table      = nullptr;

    QImage cvMatToQImage(const cv::Mat& mat);
    void   fillTable(const SliceStats& stats);
};
