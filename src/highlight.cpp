// src/highlight.cpp
#include "highlight.hpp"
#include "processing.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

// ===================== HELPERS: Operaciones Morfológicas y CC =====================
// Mantenemos las funciones de ayuda que definiste (morphCloseK, keepLargeCC, fillHoles, etc.)
// Estas funciones se definieron en el código que me pasaste.

static Mat morphOpenK(const Mat& bin, int k) {
  Mat out;
  morphologyEx(bin, out, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(k, k)));
  return out;
}

static Mat morphCloseK(const Mat& bin, int k) {
  Mat out;
  morphologyEx(bin, out, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(k, k)));
  return out;
}

static Mat morphErodeK(const Mat& bin, int k) {
  Mat out;
  erode(bin, out, getStructuringElement(MORPH_ELLIPSE, Size(k, k)));
  return out;
}

static Mat morphDilateK(const Mat& bin, int k) {
  Mat out;
  dilate(bin, out, getStructuringElement(MORPH_ELLIPSE, Size(k, k)));
  return out;
}

static Mat keepLargeCC(const Mat& bin, int minArea) {
  CV_Assert(bin.type() == CV_8U);
  Mat labels, stats, cents;
  int n = connectedComponentsWithStats(bin, labels, stats, cents, 8, CV_32S);
  Mat out = Mat::zeros(bin.size(), CV_8U);
  for (int i = 1; i < n; ++i) { 
    if (stats.at<int>(i, CC_STAT_AREA) >= minArea)
      out.setTo(255, labels == i);
  }
  return out;
}

static void fillHoles(Mat& bin) {
  CV_Assert(bin.type() == CV_8U);
  Mat inv; bitwise_not(bin, inv);
  Mat ffMask = Mat::zeros(inv.rows + 2, inv.cols + 2, CV_8U);
  Mat filled = inv.clone();
  floodFill(filled, ffMask, Point(0, 0), Scalar(0));
  Mat holes;
  subtract(inv, filled, holes); 
  bin |= holes;                 
}


// ===================== 1. Segmentación Anatómica en HU reales (CORREGIDA) =====================
AnatomyMasks generateAnatomicalMasksHU(const Mat& hu32f) {
  CV_Assert(!hu32f.empty() && hu32f.type() == CV_32F);
  AnatomyMasks M;

  // 1) Segmentación del Cuerpo (Se mantiene)
  Mat body = (hu32f > -300.f);
  body.convertTo(body, CV_8U, 255);
  body = morphCloseK(body, 7);
  fillHoles(body);

  if (countNonZero(body) < 1000) {
    M.fat = M.muscle_tendon = M.bones = Mat::zeros(hu32f.size(), CV_8U);
    return M;
  }

  // 2) Umbralización por Rangos HU (RANGOS CORREGIDOS para GRASA, MÚSCULO y HUESO)
  Mat fat, muscle, bone;
  
  // Grasa: Rango más estricto (-190 HU a -30 HU)
  fat    = (hu32f >= -190.f) & (hu32f <= -30.f);      
  // Músculo: Rango más estricto (10 HU a 120 HU)
  muscle = (hu32f >= 10.f) & (hu32f <= 120.f);        
  // Hueso: (200+ HU, incluye cortical y esponjoso)
  bone   = (hu32f > 200.f);                           

  fat.convertTo(fat, CV_8U);
  muscle.convertTo(muscle, CV_8U);
  bone.convertTo(bone, CV_8U);

  // 3) Recorte a cuerpo (Se mantiene)
  fat    &= body;
  muscle &= body;
  bone   &= body;

  // 5) Limpieza Morfológica MÍNIMA y Componentes Conexos (Se mantiene)
  bone   = morphCloseK(bone, 3);

  fat    = keepLargeCC(fat,     5); 
  muscle = keepLargeCC(muscle, 10); 
  bone   = keepLargeCC(bone,     50);

  fillHoles(muscle); 

  // 6) Lógica de Prioridad y Exclusión (CRÍTICO)
  
  // Prioridad Músculo > Grasa: Si está clasificado como músculo, no puede ser grasa.
  fat.setTo(0, muscle); 

  // Prioridad Hueso (Erosionado) > Músculo y Grasa
  // Protege el contorno del hueso
  Mat bone_erode = morphErodeK(bone, 3); 
  muscle.setTo(0, bone_erode); 
  fat.setTo(0, bone_erode);    

  // Hueso completo: Si algo es HUESO, no debe ser pintado como músculo o grasa.
  muscle.setTo(0, bone);
  fat.setTo(0, bone);


  M.fat            = fat;
  M.muscle_tendon  = muscle;
  M.bones          = bone;
  return M;
}

// ===================== 2. Highlight ROI (Función de ejemplo, se mantiene) =====================
HighlightResult highlightROI(const Mat& gray) {
  HighlightResult R;
  CV_Assert(!gray.empty());

  Mat g;
  if (gray.channels() == 3) cvtColor(gray, g, COLOR_BGR2GRAY);
  else g = gray.clone();

  threshold(g, R.mask, 28, 255, THRESH_BINARY); 
  R.mask = morphCloseK(R.mask, 9);
  fillHoles(R.mask);

  Mat baseBGR; cvtColor(g, baseBGR, COLOR_GRAY2BGR);
  R.overlay = baseBGR.clone();
  Mat color = Mat::zeros(baseBGR.size(), baseBGR.type());
  color.setTo(Scalar(0, 0, 255), R.mask); 
  addWeighted(R.overlay, 0.6, color, 0.4, 0, R.finalVis);

  return R;
}


// ===================== 3. Overlay coloreado (Alto Contraste) (Se mantiene) =====================
Mat colorizeAndOverlay(const Mat& slice8u, const AnatomyMasks& m) {
  Mat base;
  if (slice8u.channels() == 1) cvtColor(slice8u, base, COLOR_GRAY2BGR);
  else base = slice8u.clone();

  // Definición de colores BGR (Alto Contraste)
  Mat color = Mat::zeros(base.size(), base.type());
  
  // ORDEN DE PINTADO: de menor a mayor prioridad
  // 1. Grasa (Amarillo BGR)
  color.setTo(Scalar(  0, 255, 255), m.fat);           
  // 2. Músculo/Tendón (Rojo BGR)
  color.setTo(Scalar(  0,   0, 255), m.muscle_tendon); 
  // 3. Hueso (Azul BGR)
  color.setTo(Scalar(255,   0,   0), m.bones);         
  
  Mat out;
  // Resaltado fuerte (70% de máscara, 30% de fondo)
  addWeighted(base, 0.3, color, 0.7, 0, out);

  // Dibujo de Contornos (Se mantiene)
  auto drawContour = [&](const Mat& mask, const Scalar& bgr) {
    Mat edges; Canny(mask, edges, 50, 150);
    Mat dil; dilate(edges, dil, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
    out.setTo(bgr, dil);
  };
  drawContour(m.fat,           Scalar(  0, 150, 150));
  drawContour(m.muscle_tendon, Scalar(  0,   0, 150));
  drawContour(m.bones,         Scalar(150,   0,   0));

  return out;
}