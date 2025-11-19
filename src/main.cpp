// src/main.cpp - CÓDIGO FINAL Y ROBUSTO
#include "itk_loader.hpp"
#include "itk_opencv_bridge.hpp"
#include "highlight.hpp"
#include "dnn_denoising.hpp" 

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp> // Necesario para fastNlMeansDenoising

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;

// --- Damos por hecho que dnn_denoising.cpp hace la resta (input - residual) correctamente ---

int main(int argc, char** argv) {
  if (argc < 3) {
    cerr << "Uso: " << argv[0] << " <carpeta_dicom> <z_index>\n";
    return 1;
  }

  const std::string dicomDir = argv[1];
  const unsigned int zIndex  = static_cast<unsigned int>(std::stoul(argv[2]));

  DnnDenoiser* denoiserPtr = nullptr;
  
  try {
    // 1. INICIALIZACIÓN DE LA RED NEURONAL (DNN) - DENTRO DE UN BLOQUE TRY SEGURO
    try {
      // Intenta cargar el modelo de 3 capas
      denoiserPtr = new DnnDenoiser("../models/dncnn_3.onnx"); // <-- RUTA CORREGIDA
      cout << "[INFO] Modelo DnCNN cargado correctamente.\n";

    } catch (const cv::Exception& e) {
      // Capturamos el error de OpenCV, el programa NO SE CAE
      cerr << "[ADVERTENCIA] Error al cargar la red DNN: " << e.what() << "\n";
      cerr << "-> El DNN será sustituido por el método NLMeans para la evidencia de reducción avanzada.\n";
      if (denoiserPtr) { delete denoiserPtr; denoiserPtr = nullptr; }
    }
    
    // 2) Carga volumen y extrae corte ITK
    cout << "Cargando DICOM series desde: " << dicomDir << " (Slice Z=" << zIndex << ")\n";
    auto vol   = loadDicomSeries(dicomDir);
    auto slice = extractSlice(vol.image, zIndex);

    // 3) HU (float) desde ITK
    double huMin = 0.0, huMax = 0.0;
    Mat hu32f = itk2cv32fHU(slice, &huMin, &huMax);
    if (hu32f.empty()) CV_Error(Error::StsError, "hu32f vacío");
    cout << "Rango HU del slice: " << huMin << " a " << huMax << endl;

    // 4) Generar imagen 8-bit base (IMAGEN ORIGINAL CON RUIDO)
    Mat g8_base = huTo8u(hu32f, 40.0f, 400.0f);

    // ==========================================================
    // CÁLCULO DE EVIDENCIA DE REDUCCIÓN DE RUIDO
    
    // A) REDUCCIÓN CLÁSICA (Gaussiano)
    Mat g8_classical;
    GaussianBlur(g8_base, g8_classical, Size(5, 5), 0, 0); 

    // B) REDUCCIÓN AVANZADA (DNN o NLMeans como fallback)
    Mat g8_advanced;
    
    if (denoiserPtr) {
        // Opción 1: DNN cargó correctamente -> USAMOS DNN
        cout << "[INFO] Aplicando denoising con DNN...\n";
        g8_advanced = denoiserPtr->denoise(g8_base); 
    } else {
        // Opción 2: DNN falló -> USAMOS NLMeans (Sustituto avanzado)
        cout << "[INFO] Aplicando denoising con NLMeans (sustituto del DNN)...\n";
        fastNlMeansDenoising(g8_base, g8_advanced, 10, 7, 21);
    }
    // ==========================================================
    
    // Usamos la imagen con reducción de ruido avanzada para el overlay final
    Mat g8 = g8_advanced; 

    // 6) Segmentación en HU reales (Músculo/Grasa/Hueso)
    AnatomyMasks anat = generateAnatomicalMasksHU(hu32f); 
    Mat overlayAnat   = colorizeAndOverlay(g8, anat);
    
    // 5.1) Máscara de cuerpo y estadísticas (Se mantiene)
    Mat bodyMask = (hu32f > -300.f);
    bodyMask.convertTo(bodyMask, CV_8U, 255);
    const double bodyPx = (double)countNonZero(bodyMask);
    
    // 5.2) Métricas globales y conteos de depuración (Se mantiene)
    double meanHU = cv::mean(hu32f, hu32f > -1024)[0];
    const long long cntFatHU    = countNonZero( (hu32f >= -190.f) & (hu32f <=  -30.f) );
    const long long cntMuscleHU = countNonZero( (hu32f >=  10.f) & (hu32f <=  120.f) );
    const long long cntBoneHU   = countNonZero( (hu32f >=  200.f) & (hu32f <= 3000.f) );
    
    // 6) Salidas a disco 
    std::filesystem::create_directories("outputs/intermediates");
    std::filesystem::create_directories("outputs/final");
    
    // GUARDADO DE EVIDENCIA DE REDUCCIÓN DE RUIDO (Los tres archivos de comparación)
    imwrite("outputs/final/slice8u_0_original_noisy.png", g8_base); 
    imwrite("outputs/final/slice8u_1_classical_gaussian.png", g8_classical); 
    imwrite("outputs/final/slice8u_2_advanced_method.png", g8_advanced); // Puede ser DNN o NLMeans
    
    imwrite("outputs/final/overlay_anatomical_FINAL.png", overlayAnat);
    
    // 7) Estadísticos: porcentajes SOBRE EL CUERPO 
    auto pct_in_body = [&](const Mat& m){
      if (bodyPx < 1.0) return 0.0;
      Mat inter; bitwise_and(m, bodyMask, inter);
      return 100.0 * (double)countNonZero(inter) / bodyPx;
    };
    cout << "\n--- ESTADÍSTICAS SOBRE EL CUERPO ---\n";
    cout << "Áreas sobre cuerpo (%)  Grasa: " << pct_in_body(anat.fat)
         << "  | Músculo/Tendón: " << pct_in_body(anat.muscle_tendon)
         << "  | Hueso: "  << pct_in_body(anat.bones) << endl;
    cout << "------------------------------------\n";

    cout << "Guardado en outputs/ ...\n";

    imshow("Overlay anatomico (3 zonas)", overlayAnat);
    waitKey(0);

  } catch (const std::exception& e) {
    cerr << "Error: " << e.what() << "\n";
    if (denoiserPtr) { delete denoiserPtr; }
    return 2;
  }
  
  if (denoiserPtr) { delete denoiserPtr; }
  return 0;
}