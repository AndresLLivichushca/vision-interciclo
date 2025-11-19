#include "itk_loader.hpp"
#include <stdexcept>
#include <itkImageSeriesReader.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkExtractImageFilter.h>

Volume loadDicomSeries(const std::string& dicomDir) {
  auto imageIO = itk::GDCMImageIO::New();
  auto nameGen = itk::GDCMSeriesFileNames::New();
  nameGen->SetUseSeriesDetails(true);
  nameGen->SetDirectory(dicomDir);

  const auto& seriesUIDs = nameGen->GetSeriesUIDs();
  if (seriesUIDs.empty()) throw std::runtime_error("No se encontraron series DICOM en: " + dicomDir);

  const auto files = nameGen->GetFileNames(seriesUIDs.front());
  using ReaderType = itk::ImageSeriesReader<ImageType3D>;
  auto reader = ReaderType::New();
  reader->SetImageIO(imageIO);
  reader->SetFileNames(files);
  reader->Update();

  return { reader->GetOutput(), files };
}

ImageType2D::Pointer extractSlice(const ImageType3D::Pointer& vol, unsigned int z) {
  const auto region = vol->GetLargestPossibleRegion();
  const auto size   = region.GetSize();
  if (z >= size[2]) throw std::runtime_error("Índice Z fuera de rango");

  using ExtractFilter = itk::ExtractImageFilter<ImageType3D, ImageType2D>;
  auto extract = ExtractFilter::New();
  auto desired = region;
  desired.SetSize(2, 0);  // colapsar en Z
  desired.SetIndex(2, z);
  extract->SetExtractionRegion(desired);
  extract->SetDirectionCollapseToIdentity();
  extract->SetInput(vol);
  extract->Update();
  return extract->GetOutput();
}