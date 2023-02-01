#include "training.h"
#include "calc.h"
#include <stdio.h>

#define LOWER_LIM 0.2
#define UPPER_LIM 0.8

TrainingImage *generateTrainingImages(uint32_t n)
{
    TrainingImage *images = (TrainingImage*)malloc(n * sizeof(TrainingImage));

    uint8_t imageCase = 0;
    uint32_t q = (n / 4);
    for (uint32_t i = 0; i < n; i++)
    {
        double *imageValues = (double*)malloc(4 * sizeof(double));

        uint8_t elem = 0;
        double c = randomDouble(0, 1);
        switch (imageCase)
        {
            case FORM_SOLID:
                *(imageValues) = c;
                *(imageValues + 1) = c;
                *(imageValues + 2) = c;
                *(imageValues + 3) = c;
                elem = FORM_SOLID;
                break;
            case FORM_VERTICAL:
                *(imageValues) = randomDouble(UPPER_LIM, 1);
                *(imageValues + 1) = randomDouble(0, LOWER_LIM);
                *(imageValues + 2) = randomDouble(UPPER_LIM, 1);
                *(imageValues + 3) = randomDouble(0, LOWER_LIM);
                elem = FORM_VERTICAL;
                break;
            case FORM_HORIZONTAL:
                *(imageValues) = randomDouble(UPPER_LIM, 1);
                *(imageValues + 1) = randomDouble(UPPER_LIM, 1);
                *(imageValues + 2) = randomDouble(0, LOWER_LIM);
                *(imageValues + 3) = randomDouble(0, LOWER_LIM);
                elem = FORM_HORIZONTAL;
                break;
            case FORM_DIAGONAL:
                *(imageValues) = randomDouble(UPPER_LIM, 1);
                *(imageValues + 1) = randomDouble(0, LOWER_LIM);
                *(imageValues + 2) = randomDouble(0, LOWER_LIM);
                *(imageValues + 3) = randomDouble(UPPER_LIM, 1);
                elem = FORM_DIAGONAL;
                break;
        }
        if (i != 0 && i % q == 0)
        {
            imageCase++;
        }

        TrainingImage ti = {elem, imageValues};
        *(images + i) = ti;
    }

    return images;
}
