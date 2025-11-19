#include "processing.hpp"
#include <algorithm>

using namespace cv;

static Mat toGray8(const Mat& in) {
  CV_Assert(!in.empty());
  Mat g;
  if (in.channels() == 3) cvtColor(in, g, COLOR_BGR2GRAY);
  else g = in.clone();

  if (g.depth() != CV_8U) {
    double minv, maxv; minMaxLoc(g, &minv, &maxv);
    if (maxv <= minv) maxv = minv + 1.0;
    g.convertTo(g, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
  }
  return g;
}

cv::Mat equalize(const cv::Mat& g) {
  Mat gray = toGray8(g);
  Mat eq; equalizeHist(gray, eq);
  return eq;
}

cv::Mat denoiseClassic(const cv::Mat& g) {
  Mat gray = toGray8(g);
  Mat den;
  // Suaviza preservando bordes
  bilateralFilter(gray, den, /*diameter*/5, /*sigmaColor*/25, /*sigmaSpace*/7);
  return den;
}

cv::Mat edgesCanny(const cv::Mat& g, double lo, double hi) {
  Mat gray = toGray8(g);
  GaussianBlur(gray, gray, Size(3,3), 0.8);
  Mat edges;
  Canny(gray, edges, lo, hi, 3, true);
  return edges;
}

static Mat kernelEllipse(int k) {
  int kk = std::max(1, k | 1); // impar
  return getStructuringElement(MORPH_ELLIPSE, Size(kk, kk));
}

cv::Mat morphOpen(const cv::Mat& g, int k) {
  Mat bin = toGray8(g);
  threshold(bin, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
  Mat out; morphologyEx(bin, out, MORPH_OPEN, kernelEllipse(k));
  return out;
}

cv::Mat morphClose(const cv::Mat& g, int k) {
  Mat bin = toGray8(g);
  threshold(bin, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
  Mat out; morphologyEx(bin, out, MORPH_CLOSE, kernelEllipse(k));
  return out;
}