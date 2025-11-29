cmake_minimum_required(VERSION 3.16)

project(VisionInterciclo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Rutas a librerías instaladas (ajusta si cambian)
list(APPEND CMAKE_PREFIX_PATH "/opt/itk-5.4" "/opt/opencv-4.10")

# ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# OpenCV
find_package(OpenCV REQUIRED)

# ==============================
# Qt5 para la app de escritorio
# ==============================
find_package(Qt5 REQUIRED COMPONENTS Widgets)

# Que CMake genere automáticamente el MOC
set(CMAKE_AUTOMOC ON)

# Ejecutable principal (CLI + OpenCV)
add_executable(vision_interciclo
  src/main.cpp
  src/itk_loader.cpp
  src/itk_opencv_bridge.cpp
  src/processing.cpp
  src/highlight.cpp
  src/dnn_denoising.cpp
)

target_include_directories(vision_interciclo PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  src
)

target_link_libraries(vision_interciclo PRIVATE
  ${OpenCV_LIBS}
  ${ITK_LIBRARIES}
)

# ==============================
# Ejecutable Qt (ventana bonita)
# ==============================
add_executable(vision_interciclo_qt
  src/qt_main.cpp
  src/QtMainWindow.cpp
  src/CompareWindow.cpp 
  src/itk_loader.cpp
  src/itk_opencv_bridge.cpp
  src/processing.cpp
  src/highlight.cpp
  src/dnn_denoising.cpp
)

target_include_directories(vision_interciclo_qt PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  src
)

target_link_libraries(vision_interciclo_qt PRIVATE
  Qt5::Widgets
  ${OpenCV_LIBS}
  ${ITK_LIBRARIES}
)
