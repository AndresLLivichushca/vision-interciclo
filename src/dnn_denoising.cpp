// src/dnn_denoising.cpp - CÓDIGO CLAVE PARA EL APRENDIZAJE RESIDUAL

#include "dnn_denoising.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Constructor
DnnDenoiser::DnnDenoiser(const std::string& modelPath) {
    // La función readNetFromOnnx puede lanzar una excepción si el archivo es inválido
    net = dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        CV_Error(Error::StsError, "Error al cargar la red ONNX. Archivo no encontrado o inválido.");
    }
}

// Método Denoise
Mat DnnDenoiser::denoise(const Mat& noisy8u) {
    if (net.empty()) {
        // Si la red falló al cargar, retorna la imagen original
        return noisy8u.clone();
    }
    
    // 1. Preprocesamiento: Convertir CV_8U a Mat de float 0.0-1.0
    Mat input_mat;
    noisy8u.convertTo(input_mat, CV_32F, 1.0 / 255.0); // Normalizar a [0, 1]

    // 2. Preparar el blob de entrada (1x1xHxW)
    Mat blob = dnn::blobFromImage(input_mat); 

    // 3. Pasar por la red (predice el RESIDUAL, es decir, el RUIDO)
    net.setInput(blob);
    Mat residual_blob = net.forward();

    // 4. Postprocesamiento: Convertir el blob de residual a Mat
    Mat residual_mat;
    residual_blob.copyTo(residual_mat);
    residual_mat = residual_mat.reshape(1, input_mat.rows); // Remueve BATCH y CHANNEL

    // 5. APRENDIZAJE RESIDUAL (CLAVE): Imagen Limpia = Imagen Ruidosa - Ruido Predicho
    Mat output_mat;
    subtract(input_mat, residual_mat, output_mat);

    // 6. Finalizar: Recortar (clamp) a [0, 1] y convertir de vuelta a CV_8U
    output_mat.setTo(0.0f, output_mat < 0.0f);
    output_mat.setTo(1.0f, output_mat > 1.0f);
    
    Mat denoised8u;
    output_mat.convertTo(denoised8u, CV_8U, 255.0);

    return denoised8u;
}