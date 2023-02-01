#ifndef TRAINING_H
#define TRAINING_H
#include <stdlib.h>

enum {FORM_SOLID, FORM_VERTICAL, FORM_HORIZONTAL, FORM_DIAGONAL};

typedef struct
{
    uint8_t imageIndex;
    double *imageData;
} TrainingImage;

TrainingImage *generateTrainingImages(uint32_t n);
#endif
