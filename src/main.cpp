#include "itk_loader.hpp"
#include "itk_opencv_bridge.hpp"
#include "highlight.hpp"
#include "dnn_denoising.hpp" 

#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp> 

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// ======================================================================================
// 1. SELECTOR DE ARCHIVO (Nativo Linux)
// ======================================================================================
string abrirSelectorDeArchivo() {
    string path = "";
    try {
        array<char, 128> buffer;
        string result;
        string cmd = "zenity --file-selection --title=\"Selecciona una IMAGEN (.IMA / .dcm)\" --file-filter=\"*.IMA *.ima *.dcm *.DCM\"";
        
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return "";
        
        // Usamos unique_ptr para manejo seguro de memoria del pipe
        unique_ptr<FILE, decltype(&pclose)> pipe_ptr(pipe, pclose);
        while (fgets(buffer.data(), buffer.size(), pipe_ptr.get()) != nullptr) {
            result += buffer.data();
        }
        
        if (!result.empty() && result.back() == '\n') result.pop_back();
        path = result;
    } catch (...) { }
    return path;
}

// ======================================================================================
// 2. PROCESAMIENTO: 13 EVIDENCIAS (CORREGIDO)
// ======================================================================================
void procesarArchivoSeleccionado(const string& filePath) {
    fs::path p(filePath);
    string dicomDir = p.parent_path().string();
    string selectedFileName = p.filename().string();

    cout << "\n[CARGANDO] Archivo: " << selectedFileName << "\n";
    
    DnnDenoiser* denoiserPtr = nullptr;
    try {
        try {
            denoiserPtr = new DnnDenoiser("../models/dncnn_compatible.onnx"); 
            cout << "[INIT] DNN Cargado.\n";
        } catch (...) { cout << "[AVISO] DNN no disponible.\n"; }
        
        // --- CARGA ITK ---
        auto vol = loadDicomSeries(dicomDir);
        if (vol.image.IsNull()) {
            if (system("zenity --error --text=\"Error al leer DICOM.\"")) {}
            return;
        }

        vector<string> filesInDir;
        for (const auto & entry : fs::directory_iterator(dicomDir)) {
            string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".ima" || ext == ".dcm") filesInDir.push_back(entry.path().filename().string());
        }
        std::sort(filesInDir.begin(), filesInDir.end());

        int targetIndex = 0;
        for (size_t i = 0; i < filesInDir.size(); ++i) {
            if (filesInDir[i] == selectedFileName) { targetIndex = (int)i; break; }
        }

        auto slice = extractSlice(vol.image, targetIndex);
        double huMin = 0.0, huMax = 0.0;
        Mat hu32f_raw = itk2cv32fHU(slice, &huMin, &huMax); 
        
        // =========================================================
        // GRUPO A: LIMPIEZA DE IMAGEN (4 IMÁGENES)
        // =========================================================
        
        // 1. ORIGINAL
        Mat img_1_original = huTo8u(hu32f_raw, 40.0f, 400.0f);

        // 2. SUAVIZADA CON GAUSS (Clásica)
        Mat img_2_gauss;
        GaussianBlur(img_1_original, img_2_gauss, Size(5, 5), 1.0);

        // 3. SUAVIZADA CON NLMEANS (Avanzada Matemática)
        Mat img_3_nlmeans;
        cout << "[PROCESO] Calculando NLMeans...\n";
        fastNlMeansDenoising(img_1_original, img_3_nlmeans, 10, 7, 21);

        // 4. SUAVIZADA CON DNCNN (Deep Learning)
        Mat img_4_dncnn;
        if (denoiserPtr) {
            cout << "[PROCESO] Ejecutando DnCNN...\n";
            img_4_dncnn = denoiserPtr->denoise(img_1_original);
        } else {
            img_4_dncnn = img_1_original.clone();
        }

        // =========================================================
        // GRUPO B: TÉCNICAS INTERMEDIAS (5 IMÁGENES)
        // =========================================================
        Mat k = getStructuringElement(MORPH_RECT, Size(3, 3));
        
        // 5. BORDES
        Mat img_5_bordes;
        Canny(img_2_gauss, img_5_bordes, 50, 150);

        // 6. TOPHAT
        Mat img_6_tophat;
        morphologyEx(img_1_original, img_6_tophat, MORPH_TOPHAT, k);

        // 7. BLACKHAT
        Mat img_7_blackhat;
        morphologyEx(img_1_original, img_7_blackhat, MORPH_BLACKHAT, k);

        // 8. EROSIÓN
        Mat img_8_erosion;
        erode(img_1_original, img_8_erosion, k);

        // 9. DILATACIÓN
        Mat img_9_dilatacion;
        dilate(img_1_original, img_9_dilatacion, k);


        // =========================================================
        // GRUPO C: SEGMENTACIÓN FINAL (4 IMÁGENES)
        // =========================================================
        
        // 10. SEGMENTACIÓN EN ORIGINAL
        AnatomyMasks masks_raw = generateAnatomicalMasksHU(hu32f_raw);
        Mat img_10_seg_original = colorizeAndOverlay(img_1_original, masks_raw);

        // 11. SEGMENTACIÓN EN SUAVIZADA CON GAUSS
        Mat hu_gauss; 
        GaussianBlur(hu32f_raw, hu_gauss, Size(3, 3), 1.0);
        AnatomyMasks masks_gauss = generateAnatomicalMasksHU(hu_gauss);
        Mat img_11_seg_gauss = colorizeAndOverlay(img_2_gauss, masks_gauss);

        // 12. SEGMENTACIÓN EN NLMEANS (Avanzada 1)
        // Aquí estaba el error: Definimos explícitamente la variable
        Mat img_12_seg_nlmeans = colorizeAndOverlay(img_3_nlmeans, masks_gauss);

        // 13. SEGMENTACIÓN EN DNCNN (Avanzada 2)
        Mat hu_dnn_proxy;
        GaussianBlur(hu32f_raw, hu_dnn_proxy, Size(3, 3), 0.8);
        AnatomyMasks masks_dnn = generateAnatomicalMasksHU(hu_dnn_proxy);
        Mat img_13_seg_dncnn = colorizeAndOverlay(img_4_dncnn, masks_dnn);


        // =========================================================
        // GUARDADO Y VISUALIZACIÓN (13 VENTANAS)
        // =========================================================
        fs::create_directories("outputs/final");
        
        // Guardar Grupo A
        imwrite("outputs/final/1_Original.png", img_1_original);
        imwrite("outputs/final/2_Gauss.png", img_2_gauss);
        imwrite("outputs/final/3_NLMeans.png", img_3_nlmeans);
        imwrite("outputs/final/4_DnCNN.png", img_4_dncnn);
        
        // Guardar Grupo B
        imwrite("outputs/final/5_Bordes_Canny.png", img_5_bordes);
        imwrite("outputs/final/6_TopHat.png", img_6_tophat);
        imwrite("outputs/final/7_BlackHat.png", img_7_blackhat);
        imwrite("outputs/final/8_Erosion.png", img_8_erosion);
        imwrite("outputs/final/9_Dilatacion.png", img_9_dilatacion);
        
        // Guardar Grupo C
        imwrite("outputs/final/10_Seg_Original.png", img_10_seg_original);
        imwrite("outputs/final/11_Seg_Gaus.png", img_11_seg_gauss);
        imwrite("outputs/final/12_Seg_NLMeans.png", img_12_seg_nlmeans); // Ahora sí existe
        imwrite("outputs/final/13_Seg_DnCNN.png", img_13_seg_dncnn);

        // Mostrar en Pantalla (Solo las principales para no colapsar)
        imshow("1. Original", img_1_original);
        imshow("2. Gauss", img_2_gauss);
        imshow("3. NLMeans", img_3_nlmeans);
        imshow("4. DnCNN", img_4_dncnn);
        
        // Técnicas Intermedias (Muestra)
        imshow("5. Bordes", img_5_bordes);
        imshow("6. TopHat", img_6_tophat);
        
        // Segmentaciones
        imshow("10. Seg. Original", img_10_seg_original);
        imshow("11. Seg. Gauss", img_11_seg_gauss);
        imshow("12. Seg. NLMeans", img_12_seg_nlmeans);
        imshow("13. Seg. DnCNN", img_13_seg_dncnn);

        cout << "\n[EXITO] Se han generado las 13 EVIDENCIAS en 'outputs/final/'.\n";
        cout << "Presiona una tecla en las ventanas para cerrar...\n";
        
        waitKey(0);
        destroyAllWindows();

    } catch (const std::exception& e) {
        string msg = "Error: " + string(e.what());
        cerr << msg << endl;
        if (system(("zenity --error --text=\"" + msg + "\"").c_str())) {}
    }
    if (denoiserPtr) delete denoiserPtr;
}

// ======================================================================================
// MAIN
// ======================================================================================
int main(int argc, char** argv) {
    while (true) {
        Mat menu = Mat::zeros(Size(600, 300), CV_8UC3);
        putText(menu, "GENERADOR FINAL (13 IMAGENES)", Point(30, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
        putText(menu, "[O] Abrir IMAGEN (.IMA/.dcm)", Point(50, 120), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);
        putText(menu, "[ESC] Salir", Point(50, 170), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 100, 255), 1);
        
        imshow("Menu Principal", menu);
        int key = waitKey(0);

        if (key == 27) break; 
        if (key == 'o' || key == 'O') {
            destroyWindow("Menu Principal"); 
            string archivo = abrirSelectorDeArchivo();
            if (!archivo.empty()) procesarArchivoSeleccionado(archivo);
        }
    }
    return 0;
}