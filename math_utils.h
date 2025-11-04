#pragma once

#include <cmath>
#include <random>
#include "eigen_types.h"

inline double degreesToRadians(double angleDeg) {
    return angleDeg * M_PI / 180;
}

inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}