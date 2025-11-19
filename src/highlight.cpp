#include "highlight.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

// ==========================================================
// IMPLEMENTACIÓN MEJORADA: SEGMENTACIÓN SOBRE DATOS SUAVIZADOS
// ==========================================================
AnatomyMasks generateAnatomicalMasksHU(const Mat& hu32f) {
    AnatomyMasks m;

    // PASO 1: SUAVIZADO DE DATOS CRUDOS (La clave para quitar el ruido)
    // Aplicamos un filtro Gaussiano suave a los valores Hounsfield.
    // Esto elimina el "grano" del CT Scan sin perder los bordes de los órganos.
    Mat huSmooth;
    GaussianBlur(hu32f, huSmooth, Size(3, 3), 0.8); // Kernel 3x3, Sigma 0.8

    // PASO 2: UMBRALIZACIÓN (Usando la imagen suavizada)
    Mat raw_fat    = (huSmooth >= -190.f) & (huSmooth <= -30.f);
    Mat raw_muscle = (huSmooth >= 10.f)   & (huSmooth <= 120.f);
    Mat raw_bone   = (huSmooth >= 200.f);

    // Convertir a 8-bit
    raw_fat.convertTo(m.fat, CV_8U, 255);
    raw_muscle.convertTo(m.muscle_tendon, CV_8U, 255);
    raw_bone.convertTo(m.bones, CV_8U, 255);

    // PASO 3: LIMPIEZA MORFOLÓGICA (Rellena huecos y une áreas)
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat kernelLg = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    // Hueso: Cierre para que se vea sólido
    morphologyEx(m.bones, m.bones, MORPH_CLOSE, kernel);

    // Músculo: Apertura (quita puntos aislados) + Cierre (une fibras musculares)
    morphologyEx(m.muscle_tendon, m.muscle_tendon, MORPH_OPEN, kernel);
    morphologyEx(m.muscle_tendon, m.muscle_tendon, MORPH_CLOSE, kernelLg);

    // Grasa: Apertura
    morphologyEx(m.fat, m.fat, MORPH_OPEN, kernel);

    // PASO 4: JERARQUÍA (El hueso manda sobre el músculo)
    m.muscle_tendon.setTo(0, m.bones);
    m.fat.setTo(0, m.muscle_tendon);
    m.fat.setTo(0, m.bones);

    return m;
}


// ==========================================================
// IMPLEMENTACIÓN DE colorizeAndOverlay (COLORES VIVOS)
// ==========================================================
Mat colorizeAndOverlay(const Mat& slice8u, const AnatomyMasks& m) {
    Mat base_bgr;
    if (slice8u.channels() == 1) cvtColor(slice8u, base_bgr, COLOR_GRAY2BGR); 
    else base_bgr = slice8u.clone();

    // --- COLORES TIPO ATLAS DE ANATOMÍA (Vivos pero profesionales) ---
    
    // Hueso: AZUL CELESTE (Cyan) - Resalta mucho sobre el rojo oscuro
    const Scalar COLOR_BONE = Scalar(255, 255, 0); // BGR
    
    // Músculo: ROJO CARMESÍ / MAGENTA - Diferente al fondo grisáceo
    const Scalar COLOR_MUSCLE = Scalar(100, 0, 200); 
    
    // Grasa: AMARILLO / DORADO
    const Scalar COLOR_FAT = Scalar(0, 200, 255); 

    // Crear capas
    Mat colored_fat = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_fat.setTo(COLOR_FAT, m.fat);

    Mat colored_muscle = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_muscle.setTo(COLOR_MUSCLE, m.muscle_tendon);

    Mat colored_bone = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_bone.setTo(COLOR_BONE, m.bones);

    Mat final_overlay = base_bgr.clone();

    // Transparencia Alta (Para que parezca pintado pero deje ver textura)
    double alpha = 0.55; 

    if (countNonZero(m.fat) > 0)
        addWeighted(final_overlay, 1.0, colored_fat, alpha, 0, final_overlay);
        
    if (countNonZero(m.muscle_tendon) > 0)
        addWeighted(final_overlay, 1.0, colored_muscle, alpha, 0, final_overlay);
        
    if (countNonZero(m.bones) > 0)
        addWeighted(final_overlay, 1.0, colored_bone, alpha, 0, final_overlay);

    return final_overlay;
}