#ifndef CALC_H
#define CALC_H

#include <stdlib.h>
#include <math.h>
#define E 2.71828182845904
double Sigmoid(double value);
double randomDouble(double fMin, double fMax);
uint8_t highestElementIndex(double *arr, uint8_t *size);
double *expectedOutput(uint8_t *index, uint8_t *size);
#endif
