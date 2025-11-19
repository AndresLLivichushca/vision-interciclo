#pragma once
#include "itk_loader.hpp"      // define ImageType2D
#include <opencv2/core.hpp>

// Convierte un slice ITK (short HU con MetaData) a CV_32F en HU reales.
// Devuelve además min/max HU si se pasan punteros.
cv::Mat itk2cv32fHU(ImageType2D::Pointer slice,
                    double* outMinHU = nullptr,
                    double* outMaxHU = nullptr);

// Ventaneo HU → 8-bit para visualización (center/width estilo DICOM)
cv::Mat huTo8u(const cv::Mat& hu32f, float center, float width);