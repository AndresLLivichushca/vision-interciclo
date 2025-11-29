#pragma once

// Estructura para estadísticas de HU

struct TissueStats {
    double meanHU     = 0.0;
    double stdHU      = 0.0;
    int    pixelCount = 0;
};

struct SliceStats {
    TissueStats fat;      // grasa
    TissueStats muscle;   // músculo/ tendon 
    TissueStats bone;     // hueso
};
