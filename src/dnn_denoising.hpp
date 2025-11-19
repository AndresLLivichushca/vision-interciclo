#ifndef DNN_DENOISER_HPP
#define DNN_DENOISER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <iostream>

class DnnDenoiser {
public:
    // Constructor: Carga el modelo ONNX desde la ruta especificada
    DnnDenoiser(const std::string& modelPath);

    // Método principal para limpiar la imagen
    // input: Imagen en escala de grises (CV_8U o CV_32F)
    // return: Imagen limpia (denoised)
    cv::Mat denoise(const cv::Mat& inputImage);

private:
    cv::dnn::Net net;
    bool modelLoaded;
};

#endif // DNN_DENOISER_HPP