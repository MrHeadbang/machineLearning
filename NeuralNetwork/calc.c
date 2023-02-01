#include "calc.h"
double Sigmoid(double value)
{
    //return tanh(value);
    return 1 / (1 + pow(E, -value));
}

double randomDouble(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

uint8_t highestElementIndex(double *arr, uint8_t *size)
{
    double highestValue = 0;
    int highestIndex = 0;

    for (uint8_t i = 0; i < *size; i++)
    {
        if (*(arr + i) > highestValue)
        {
            highestIndex = i;
            highestValue = *(arr + i);
        }
    }

    return highestIndex;
}

double *expectedOutput(uint8_t *index, uint8_t *size)
{
    double *output = (double*)malloc(*size * sizeof(double));
    for (uint8_t i = 0; i < *size; i++)
    {
        if (i == *index)
        {
            *(output + i) = 1;
            continue;
        }
        *(output + i) = 0; 
    }
    return output;
}
