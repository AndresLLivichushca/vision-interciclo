#include "CompareWindow.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QHeaderView>

#include "Stats.hpp"   // Usa TissueStats y SliceStats
#include <opencv2/imgproc.hpp>

using namespace cv;

// =============================
// cv::Mat -> QImage
// =============================
QImage CompareWindow::cvMatToQImage(const cv::Mat& mat) {
    if (mat.empty()) return {};

    if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step,
                      QImage::Format_Grayscale8).copy();
    } else if (mat.type() == CV_8UC3) {
        Mat rgb;
        cvtColor(mat, rgb, COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step,
                      QImage::Format_RGB888).copy();
    }

    Mat tmp;
    mat.convertTo(tmp, CV_8U);
    return QImage(tmp.data, tmp.cols, tmp.rows, tmp.step,
                  QImage::Format_Grayscale8).copy();
}

// =============================
// Constructor
// =============================
CompareWindow::CompareWindow(QWidget* parent,
                             const cv::Mat& origGray,
                             const cv::Mat& gaussGray,
                             const cv::Mat& nlGray,
                             const cv::Mat& dncnnGray,
                             const cv::Mat& overlayOrig,
                             const cv::Mat& overlayGauss,
                             const cv::Mat& overlayNL,
                             const cv::Mat& overlayDncnn,
                             const SliceStats& stats)
    : QDialog(parent)
{
    setWindowTitle("Comparativa de resultados – Gauss vs NLMeans vs DnCNN");
    resize(1600, 800);

    auto* mainLayout    = new QVBoxLayout(this);
    auto* columnsLayout = new QHBoxLayout();

    // Helper para crear una columna: título + overlay + gris
    auto makeColumn = [&](const QString& title,
                          const cv::Mat& overlay,
                          const cv::Mat& gray,
                          QLabel*& outOverlayLbl,
                          QLabel*& outGrayLbl)
    {
        auto* colWidget = new QWidget(this);
        auto* vLayout   = new QVBoxLayout(colWidget);

        auto* titleLbl = new QLabel(title, colWidget);
        titleLbl->setAlignment(Qt::AlignCenter);

        // Scroll para overlay (color)
        auto* scrollOverlay = new QScrollArea(colWidget);
        outOverlayLbl = new QLabel(scrollOverlay);
        outOverlayLbl->setAlignment(Qt::AlignCenter);
        outOverlayLbl->setMinimumSize(256, 256);
        outOverlayLbl->setScaledContents(true);
        scrollOverlay->setWidget(outOverlayLbl);
        scrollOverlay->setWidgetResizable(true);

        // Scroll para gris
        auto* scrollGray = new QScrollArea(colWidget);
        outGrayLbl = new QLabel(scrollGray);
        outGrayLbl->setAlignment(Qt::AlignCenter);
        outGrayLbl->setMinimumSize(256, 256);
        outGrayLbl->setScaledContents(true);
        scrollGray->setWidget(outGrayLbl);
        scrollGray->setWidgetResizable(true);

        vLayout->addWidget(titleLbl);
        vLayout->addWidget(scrollOverlay);
        vLayout->addWidget(scrollGray);

        columnsLayout->addWidget(colWidget);

        // Asignar imágenes
        outOverlayLbl->setPixmap(QPixmap::fromImage(cvMatToQImage(overlay)));
        outGrayLbl->setPixmap(QPixmap::fromImage(cvMatToQImage(gray)));
    };

    // Columna 0: Original
    makeColumn("Original",
               overlayOrig,
               origGray,
               m_lblOverlayOrig,
               m_lblGrayOrig);

    // Columna 1: Gaussiano
    makeColumn("Filtro Gaussiano",
               overlayGauss,
               gaussGray,
               m_lblOverlayGauss,
               m_lblGrayGauss);

    // Columna 2: NLMeans
    makeColumn("Filtro NLMeans",
               overlayNL,
               nlGray,
               m_lblOverlayNL,
               m_lblGrayNL);

    // Columna 3: DnCNN
    makeColumn("Red DnCNN",
               overlayDncnn,
               dncnnGray,
               m_lblOverlayDncnn,
               m_lblGrayDncnn);

    mainLayout->addLayout(columnsLayout);

    // =============================
    // Tabla de HU por tejido
    // =============================
    m_table = new QTableWidget(3, 3, this); // 3 tejidos x 3 columnas
    QStringList colHeaders;
    colHeaders << "Media HU" << "Desv. estándar" << "Nº píxeles";
    m_table->setHorizontalHeaderLabels(colHeaders);

    QStringList rowHeaders;
    rowHeaders << "Grasa" << "Músculo / Tendón" << "Hueso";
    m_table->setVerticalHeaderLabels(rowHeaders);

    fillTable(stats);

    m_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_table->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);

    mainLayout->addWidget(m_table);
}

// =============================
// Tabla Comparativa
// =============================
void CompareWindow::fillTable(const SliceStats& stats)
{
    auto setRow = [&](int row, const TissueStats& t)
    {
        m_table->setItem(row, 0,
            new QTableWidgetItem(QString::number(t.meanHU, 'f', 1)));
        m_table->setItem(row, 1,
            new QTableWidgetItem(QString::number(t.stdHU,  'f', 1)));
        m_table->setItem(row, 2,
            new QTableWidgetItem(QString::number(t.pixelCount)));
    };

    setRow(0, stats.fat);
    setRow(1, stats.muscle);
    setRow(2, stats.bone);
}
