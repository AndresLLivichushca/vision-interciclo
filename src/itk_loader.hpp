#pragma once
#include <string>
#include <vector>
#include <itkImage.h>

using PixelType = signed short;
constexpr unsigned int Dimension = 3;
using ImageType3D = itk::Image< PixelType, Dimension >;
using ImageType2D = itk::Image< PixelType, 2 >;

struct Volume {
  ImageType3D::Pointer image;
  std::vector<std::string> files;
};

Volume loadDicomSeries(const std::string& dicomDir);
ImageType2D::Pointer extractSlice(const ImageType3D::Pointer& vol, unsigned int indexZ);