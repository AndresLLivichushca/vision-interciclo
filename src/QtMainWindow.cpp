#include "QtMainWindow.hpp"

#include "itk_loader.hpp"
#include "itk_opencv_bridge.hpp"
#include "highlight.hpp"
#include "dnn_denoising.hpp"
#include "CompareWindow.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollArea>
#include <QDir>

#include <filesystem>
#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

namespace fs = std::filesystem;
using namespace cv;

// ----------------------
// Helper para stats HU
// ----------------------
static TissueStats computeTissueStats(const cv::Mat& hu32f, const cv::Mat& mask8u) {
    TissueStats ts;
    if (hu32f.empty() || mask8u.empty()) return ts;

    cv::Mat validMask;
    // el mask está en 0/255 → lo pasamos a binario
    threshold(mask8u, validMask, 0, 255, THRESH_BINARY);

    ts.pixelCount = countNonZero(validMask);
    if (ts.pixelCount == 0) return ts;

    cv::Scalar mean, stddev;
    meanStdDev(hu32f, mean, stddev, validMask);
    ts.meanHU = mean[0];
    ts.stdHU  = stddev[0];

    return ts;
}

// ============================
// Constructor
// ============================
QtMainWindow::QtMainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("Vision Interciclo - Aplicación de escritorio");

    auto* central     = new QWidget(this);
    auto* mainLayout  = new QVBoxLayout(central);

    // -------- Fila superior: selector + botones --------
    auto* topLayout   = new QHBoxLayout();

    m_pathEdit        = new QLineEdit(central);
    m_browseButton    = new QPushButton("Seleccionar imagen DICOM", central);
    m_processButton   = new QPushButton("Procesar y mostrar", central);
    m_compareButton   = new QPushButton("Realizar comparativa", central);
    m_viewCombo       = new QComboBox(central);

    // Items del combo (evidencias)
    m_viewCombo->addItem("1. Original");
    m_viewCombo->addItem("2. Suavizada Gauss");
    m_viewCombo->addItem("3. Suavizada NLMeans");
    m_viewCombo->addItem("4. Suavizada DnCNN");
    m_viewCombo->addItem("5. Bordes Canny");
    m_viewCombo->addItem("6. TopHat");
    m_viewCombo->addItem("7. BlackHat");
    m_viewCombo->addItem("8. Erosión");
    m_viewCombo->addItem("9. Dilatación");
    m_viewCombo->addItem("10. Seg. Original");
    m_viewCombo->addItem("11. Seg. Gauss");
    m_viewCombo->addItem("12. Seg. NLMeans");   
    m_viewCombo->addItem("13. Seg. DnCNN");  
    m_viewCombo->setCurrentIndex(12); 


    // Orden: ruta | seleccionar | procesar | comparativa | combo
    topLayout->addWidget(m_pathEdit);
    topLayout->addWidget(m_browseButton);
    topLayout->addWidget(m_processButton);
    topLayout->addWidget(m_compareButton);
    topLayout->addWidget(m_viewCombo);

    mainLayout->addLayout(topLayout);

    // -------- Zona de imágenes --------
    auto* imagesLayout = new QHBoxLayout();

    auto* scrollLeft = new QScrollArea(central);
    m_originalLabel = new QLabel("Original", scrollLeft);
    m_originalLabel->setAlignment(Qt::AlignCenter);
    m_originalLabel->setMinimumSize(256, 256);
    m_originalLabel->setScaledContents(true);
    scrollLeft->setWidget(m_originalLabel);
    scrollLeft->setWidgetResizable(true);

    auto* scrollRight = new QScrollArea(central);
    m_resultLabel = new QLabel("Resultado", scrollRight);
    m_resultLabel->setAlignment(Qt::AlignCenter);
    m_resultLabel->setMinimumSize(256, 256);
    m_resultLabel->setScaledContents(true);
    scrollRight->setWidget(m_resultLabel);
    scrollRight->setWidgetResizable(true);

    imagesLayout->addWidget(scrollLeft);
    imagesLayout->addWidget(scrollRight);

    mainLayout->addLayout(imagesLayout);

    setCentralWidget(central);

    // -------- Conexiones --------
    connect(m_browseButton,  &QPushButton::clicked,
            this,            &QtMainWindow::onBrowseDicom);
    connect(m_processButton, &QPushButton::clicked,
            this,            &QtMainWindow::onProcess);
    connect(m_viewCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &QtMainWindow::onViewChanged);
    connect(m_compareButton, &QPushButton::clicked,
            this,            &QtMainWindow::onOpenCompare);
}

// ============================
// Slots
// ============================
void QtMainWindow::onBrowseDicom() {
    QString filePath = QFileDialog::getOpenFileName(
        this,
        tr("Selecciona una imagen DICOM (.IMA / .dcm)"),
        "/home/andres/Documents/interciclo/data/CT_low_dose_reconstruction_dataset/Original Data/Full Dose/3mm Slice Thickness/Sharp Kernel (D45)/L096/full_3mm_sharp",
        tr("DICOM (*.IMA *.ima *.dcm *.DCM)")
    );

    if (!filePath.isEmpty()) {
        m_pathEdit->setText(filePath);
    }
}

void QtMainWindow::onProcess() {
    QString filePath = m_pathEdit->text().trimmed();
    if (filePath.isEmpty()) {
        QMessageBox::warning(this, "Advertencia", "Selecciona primero una imagen DICOM.");
        return;
    }

    try {
        runPipelineForFile(filePath, m_results, m_stats);
        m_hasResults = true;
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", e.what());
        m_hasResults = false;
        return;
    }

    // Mostrar original a la izquierda
    QImage imgOrig = cvMatToQImage(m_results[0]);
    m_originalLabel->setPixmap(QPixmap::fromImage(imgOrig));

    // Panel derecho según el combo
    refreshResultView();
}

void QtMainWindow::onViewChanged(int) {
    if (!m_hasResults) return;
    refreshResultView();
}

void QtMainWindow::onOpenCompare() {
    if (!m_hasResults) {
        QMessageBox::information(this, "Información",
                                 "Primero procesa una imagen para poder comparar.");
        return;
    }

    CompareWindow dlg(
        this,
        m_results[0],   // origGray
        m_results[1],   // gaussGray
        m_results[2],   // nlGray
        m_results[3],   // dncnnGray
        m_results[9],   // overlayOrig
        m_results[10],  // overlayGauss
        m_results[11],  // overlayNLMeans
        m_results[12],  // overlayDncnn
        m_stats
    );
    dlg.exec();
}

// ============================
// cv::Mat -> QImage
// ============================
QImage QtMainWindow::cvMatToQImage(const cv::Mat& mat) {
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

// ============================
// Actualizar panel derecho
// ============================
void QtMainWindow::refreshResultView() {
    int idx = m_viewCombo->currentIndex();
    if (idx < 0 || idx >= static_cast<int>(m_results.size())) return;

    const Mat& m = m_results[idx];
    QImage qimg = cvMatToQImage(m);
    m_resultLabel->setPixmap(QPixmap::fromImage(qimg));
}

// ============================
// Pipeline de imagenes
// ============================
void QtMainWindow::runPipelineForFile(
        const QString& qFilePath,
        std::array<cv::Mat, 13>& outResults,
        SliceStats& outStats)
{
    std::string filePath = qFilePath.toStdString();
    fs::path p(filePath);

    std::string dicomDir         = p.parent_path().string();
    std::string selectedFileName = p.filename().string();

    // --- Carga serie ITK ---
    auto vol = loadDicomSeries(dicomDir);
    if (vol.image.IsNull()) {
        throw std::runtime_error("Error al leer la serie DICOM.");
    }

    // Encontrar índice del archivo seleccionado
    std::vector<std::string> filesInDir;
    for (const auto & entry : fs::directory_iterator(dicomDir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".ima" || ext == ".dcm") {
            filesInDir.push_back(entry.path().filename().string());
        }
    }
    std::sort(filesInDir.begin(), filesInDir.end());

    int targetIndex = 0;
    for (size_t i = 0; i < filesInDir.size(); ++i) {
        if (filesInDir[i] == selectedFileName) {
            targetIndex = static_cast<int>(i);
            break;
        }
    }

    auto slice = extractSlice(vol.image, targetIndex);

    // Convertir a HU (float)
    double huMin = 0.0, huMax = 0.0;
    Mat hu32f_raw = itk2cv32fHU(slice, &huMin, &huMax);

    // ================== GRUPO A: LIMPIEZA ==================
    Mat img_1_original = huTo8u(hu32f_raw, 40.0f, 400.0f);    // 1

    Mat img_2_gauss;                                         // 2
    GaussianBlur(img_1_original, img_2_gauss, Size(5, 5), 1.0);

    Mat img_3_nlmeans;                                       // 3
    fastNlMeansDenoising(img_1_original, img_3_nlmeans, 10, 7, 21);

    Mat img_4_dncnn;                                         // 4
    try {
        DnnDenoiser denoiser("../models/dncnn_compatible.onnx");
        img_4_dncnn = denoiser.denoise(img_1_original);
    } catch (...) {
        img_4_dncnn = img_1_original.clone();
    }

    // ========== GRUPO B: MORFOLOGÍA + BORDES ==============
    Mat k = getStructuringElement(MORPH_RECT, Size(3, 3));

    Mat img_5_bordes;                                        // 5
    Canny(img_2_gauss, img_5_bordes, 50, 150);

    Mat img_6_tophat;                                        // 6
    morphologyEx(img_1_original, img_6_tophat, MORPH_TOPHAT, k);

    Mat img_7_blackhat;                                      // 7
    morphologyEx(img_1_original, img_7_blackhat, MORPH_BLACKHAT, k);

    Mat img_8_erosion;                                       // 8
    erode(img_1_original, img_8_erosion, k);

    Mat img_9_dilatacion;                                    // 9
    dilate(img_1_original, img_9_dilatacion, k);

    // ================== GRUPO C: SEGMENTACIÓN =============

// 10. Seg. en original
AnatomyMasks masks_raw = generateAnatomicalMasksHU(hu32f_raw);
Mat img_10_seg_original = colorizeAndOverlay(img_1_original, masks_raw);

// 11. Seg. en Gauss (segmentación clásica)
Mat hu_gauss;
GaussianBlur(hu32f_raw, hu_gauss, Size(3, 3), 1.0);
AnatomyMasks masks_gauss = generateAnatomicalMasksHU(hu_gauss);
Mat img_11_seg_gauss = colorizeAndOverlay(img_2_gauss, masks_gauss);

// 12. Seg. en NLMeans (usamos las mismas máscaras que Gauss)
Mat img_12_seg_nlmeans = colorizeAndOverlay(img_3_nlmeans, masks_gauss);

// 13. Seg. en DnCNN (segmentación avanzada)
Mat hu_dnn_proxy;
GaussianBlur(hu32f_raw, hu_dnn_proxy, Size(3, 3), 0.8);
AnatomyMasks masks_dnn = generateAnatomicalMasksHU(hu_dnn_proxy);
Mat img_13_seg_dncnn = colorizeAndOverlay(img_4_dncnn, masks_dnn);

// ================== Estadísticas HU (igual que antes) =================
outStats.fat    = computeTissueStats(hu32f_raw, masks_raw.fat);
outStats.muscle = computeTissueStats(hu32f_raw, masks_raw.muscle_tendon);
outStats.bone   = computeTissueStats(hu32f_raw, masks_raw.bones);

// Guardado en disco
fs::create_directories("outputs/final_qt");
imwrite("outputs/final_qt/1_Original.png",          img_1_original);
imwrite("outputs/final_qt/2_Suavizada_Gaus.png",    img_2_gauss);
imwrite("outputs/final_qt/3_Suavizada_NLMeans.png", img_3_nlmeans);
imwrite("outputs/final_qt/4_Suavizada_DnCNN.png",   img_4_dncnn);
imwrite("outputs/final_qt/5_Bordes_Canny.png",      img_5_bordes);
imwrite("outputs/final_qt/6_TopHat.png",            img_6_tophat);
imwrite("outputs/final_qt/7_BlackHat.png",          img_7_blackhat);
imwrite("outputs/final_qt/8_Erosion.png",           img_8_erosion);
imwrite("outputs/final_qt/9_Dilatacion.png",        img_9_dilatacion);
imwrite("outputs/final_qt/10_Seg_Original.png",     img_10_seg_original);
imwrite("outputs/final_qt/11_Seg_Gaus.png",         img_11_seg_gauss);
imwrite("outputs/final_qt/12_Seg_NLMeans.png",      img_12_seg_nlmeans);
imwrite("outputs/final_qt/13_Seg_DnCNN.png",        img_13_seg_dncnn);

// Copiar a arreglo de salida (OJO con los índices)
outResults[0]  = img_1_original;
outResults[1]  = img_2_gauss;
outResults[2]  = img_3_nlmeans;
outResults[3]  = img_4_dncnn;
outResults[4]  = img_5_bordes;
outResults[5]  = img_6_tophat;
outResults[6]  = img_7_blackhat;
outResults[7]  = img_8_erosion;
outResults[8]  = img_9_dilatacion;
outResults[9]  = img_10_seg_original;
outResults[10] = img_11_seg_gauss;
outResults[11] = img_12_seg_nlmeans; // NUEVO
outResults[12] = img_13_seg_dncnn;   // DnCNN ahora es índice 12
}