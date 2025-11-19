#include "itk_opencv_bridge.hpp"
#include <itkImageRegionConstIterator.h>
#include <algorithm>

using namespace cv;

// Helper: copia ITK (short) a CV_32F en HU tal cual
cv::Mat itk2cv32fHU(ImageType2D::Pointer slice, double* outMinHU, double* outMaxHU) {
  auto region = slice->GetLargestPossibleRegion();
  auto size   = region.GetSize(); // [x,y]

  Mat hu(size[1], size[0], CV_32F);

  itk::ImageRegionConstIterator<ImageType2D> it(slice, region);
  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();

  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
    const auto& idx = it.GetIndex(); // [x,y]
    float v = static_cast<float>(it.Get()); // HU (assumiendo DICOM ya en HU)
    hu.at<float>(idx[1], idx[0]) = v;
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }

  if (outMinHU) *outMinHU = mn;
  if (outMaxHU) *outMaxHU = mx;
  return hu;
}

// Windowing HU -> 8U
cv::Mat huTo8u(const cv::Mat& hu32f, float center, float width) {
  CV_Assert(hu32f.type() == CV_32F);
  const float low  = center - width * 0.5f;
  const float high = center + width * 0.5f;

  Mat out;
  // (hu - low) / (high-low) * 255
  hu32f.convertTo(out, CV_32F, 255.0f / std::max(1e-6f, (high - low)), -low * 255.0f / std::max(1e-6f, (high - low)));
  // saturar a [0,255] y pasar a 8U
  cv::min(out, 255, out);
  cv::max(out, 0, out);
  out.convertTo(out, CV_8U);
  return out;
}