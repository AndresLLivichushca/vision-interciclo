#pragma once
#include <opencv2/opencv.hpp>

// Preprocesado base
cv::Mat equalize(const cv::Mat& g);
cv::Mat denoiseClassic(const cv::Mat& g);
cv::Mat edgesCanny(const cv::Mat& g, double lo = 40, double hi = 120);

// Morfología
cv::Mat morphOpen(const cv::Mat& g, int k = 3);
cv::Mat morphClose(const cv::Mat& g, int k = 3);