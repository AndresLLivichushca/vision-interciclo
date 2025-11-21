#include "itk_loader.hpp"
#include "itk_opencv_bridge.hpp"
#include "highlight.hpp"
#include "dnn_denoising.hpp" 

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp> 

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv) {
  if (argc < 3) {
    cerr << "Uso: " << argv[0] << " <carpeta_dicom> <z_index>\n";
    return 1;
  }

  const std::string dicomDir = argv[1];
  const unsigned int zIndex  = static_cast<unsigned int>(std::stoul(argv[2]));

  // Puntero seguro para la red neuronal
  DnnDenoiser* denoiserPtr = nullptr;
  
  try {
    // =========================================================
    // 0. INICIALIZACIÓN Y CARGA DE DNN
    // =========================================================
    try {
        denoiserPtr = new DnnDenoiser("../models/dncnn_compatible.onnx"); 
        cout << "[INIT] Modelo DnCNN cargado correctamente.\n";
    } catch (const cv::Exception& e) {
        cerr << "[ALERTA] Fallo carga DNN (" << e.what() << "). Se usara NLMeans.\n";
    }
    
    // =========================================================
    // 1. PREPARACIÓN DE DATOS
    // =========================================================
    cout << "[DATA] Cargando DICOM...\n";
    auto vol   = loadDicomSeries(dicomDir);
    auto slice = extractSlice(vol.image, zIndex);
    
    double huMin = 0.0, huMax = 0.0;
    Mat hu32f_raw = itk2cv32fHU(slice, &huMin, &huMax); 
    if (hu32f_raw.empty()) CV_Error(Error::StsError, "hu32f vacio");

    // IMAGEN VISUAL BASE (Contrast Stretching / Windowing)
    Mat img_visual_raw = huTo8u(hu32f_raw, 40.0f, 400.0f); 

    // Crear directorios de salida
    std::filesystem::create_directories("outputs/final");
    std::filesystem::create_directories("outputs/intermediates");


    // =========================================================
    // 2. GENERACIÓN DE EVIDENCIAS TÉCNICAS (RÚBRICA)
    // =========================================================
    cout << "[RÚBRICA] Generando evidencias técnicas intermedias...\n";

    // A. SUAVIZADO (Filtro Gaussiano)
    Mat img_smooth;
    GaussianBlur(img_visual_raw, img_smooth, Size(5, 5), 1.0);
    imwrite("outputs/intermediates/Tecnica_Suavizado_Gauss.png", img_smooth);

    // B. DETECCIÓN DE BORDES (Canny)
    Mat img_edges;
    Canny(img_smooth, img_edges, 50, 150);
    imwrite("outputs/intermediates/Tecnica_Bordes_Canny.png", img_edges);

    // C. OPERACIONES MORFOLÓGICAS (Detalladas)
    // Usamos un kernel visible para que el efecto sea claro en el reporte
    Mat kernel_morph = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat kernel_hat   = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));

    Mat img_erode, img_dilate, img_tophat, img_blackhat;

    // C.1 Erosión
    erode(img_visual_raw, img_erode, kernel_morph);
    imwrite("outputs/intermediates/Tecnica_Morf_Erosion.png", img_erode);

    // C.2 Dilatación
    dilate(img_visual_raw, img_dilate, kernel_morph);
    imwrite("outputs/intermediates/Tecnica_Morf_Dilatacion.png", img_dilate);

    // C.3 TopHat (Resalta detalles brillantes pequeños)
    morphologyEx(img_visual_raw, img_tophat, MORPH_TOPHAT, kernel_hat);
    imwrite("outputs/intermediates/Tecnica_Morf_TopHat.png", img_tophat);

    // C.4 BlackHat (Resalta detalles oscuros pequeños)
    morphologyEx(img_visual_raw, img_blackhat, MORPH_BLACKHAT, kernel_hat);
    imwrite("outputs/intermediates/Tecnica_Morf_BlackHat.png", img_blackhat);


    // =========================================================
    // 3. PROCESAMIENTO COMPARATIVO (3 FLUJOS)
    // =========================================================
    
    // --- FLUJO 1: ORIGINAL (CONTROL) ---
    // Datos crudos -> Segmentación sucia
    AnatomyMasks masks_raw = generateAnatomicalMasksHU(hu32f_raw);
    Mat overlay_raw = colorizeAndOverlay(img_visual_raw, masks_raw);

    // --- FLUJO 2: CLÁSICO (GAUSSIANO) ---
    // Datos suavizados -> Segmentación limpia estándar
    Mat hu32f_gauss;
    GaussianBlur(hu32f_raw, hu32f_gauss, Size(3, 3), 0.8); // Suavizado médico
    AnatomyMasks masks_gauss = generateAnatomicalMasksHU(hu32f_gauss);
    Mat overlay_gauss = colorizeAndOverlay(img_smooth, masks_gauss);

    // --- FLUJO 3: AVANZADO (DNN / NLMeans) ---
    Mat img_visual_dnn;
    Mat hu32f_dnn;

    if (denoiserPtr) {
        cout << "[PROCESO] Usando DNN (DnCNN)...\n";
        // Limpieza Visual
        img_visual_dnn = denoiserPtr->denoise(img_visual_raw);
        
        // Limpieza de Datos Médicos (Normalización temporal)
        Mat hu_norm;
        normalize(hu32f_raw, hu_norm, 0, 255, NORM_MINMAX, CV_8U);
        Mat hu_clean_8u = denoiserPtr->denoise(hu_norm);
        hu_clean_8u.convertTo(hu32f_dnn, CV_32F);
        normalize(hu32f_dnn, hu32f_dnn, huMin, huMax, NORM_MINMAX);
    } else {
        cout << "[PROCESO] Usando NLMeans (Fallback Avanzado)...\n";
        // Limpieza Visual
        fastNlMeansDenoising(img_visual_raw, img_visual_dnn, 10, 7, 21);
        // Limpieza Datos Médicos (Usamos Gaussiano fino como proxy de alta calidad)
        GaussianBlur(hu32f_raw, hu32f_dnn, Size(3, 3), 0.8); 
    }

    // Generamos las máscaras avanzadas y el overlay final
    AnatomyMasks masks_dnn = generateAnatomicalMasksHU(hu32f_dnn);
    // Usamos la imagen visualmente limpia por DNN/NLMeans de fondo
    Mat overlay_dnn = colorizeAndOverlay(img_visual_dnn, masks_dnn); 


    // =========================================================
    // 4. GUARDADO DE COMPARATIVAS FINALES
    // =========================================================
    
    // GRUPO 1: IMÁGENES SIN SEGMENTACIÓN (Evidencia de reducción de ruido)
    imwrite("outputs/final/1_Ruido_Original.png", img_visual_raw);
    imwrite("outputs/final/2_Ruido_Clasico.png", img_smooth);
    imwrite("outputs/final/3_Ruido_Avanzado.png", img_visual_dnn); // DNN o NLMeans

    // GRUPO 2: IMÁGENES CON SEGMENTACIÓN (Evidencia de mejora de análisis)
    imwrite("outputs/final/4_Segm_Original_Mala.png", overlay_raw);
    imwrite("outputs/final/5_Segm_Clasica.png", overlay_gauss);
    imwrite("outputs/final/6_Segm_Avanzada_Final.png", overlay_dnn);

    cout << "\n========================================================\n";
    cout << "¡PROCESO COMPLETADO!\n";
    cout << "Revisa 'outputs/intermediates/' para ver Erosion, Dilatacion, Bordes, etc.\n";
    cout << "Revisa 'outputs/final/' para ver las 6 imagenes comparativas.\n";
    cout << "========================================================\n";

    // Mostrar resultado final en pantalla
    imshow("Resultado Final (Avanzado)", overlay_dnn);
    waitKey(0);

  } catch (const std::exception& e) {
    cerr << "Error Fatal: " << e.what() << "\n";
    if (denoiserPtr) delete denoiserPtr;
    return 2;
  }
  
  if (denoiserPtr) delete denoiserPtr;
  return 0;
}