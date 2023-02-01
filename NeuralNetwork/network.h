#ifndef NETWORK_H
#define NETWORK_H
#include "training.h"

typedef struct
{
    uint8_t layersNum;
    uint8_t *neuronNum;
} NetworkInfo;

typedef struct
{
    double ***weights;
    double **biases;
} NetworkData;

double ***initializeWeights(NetworkInfo *netinf);
double **initializeBiases(NetworkInfo *netinf);
double *forwardPass(NetworkInfo *netinf, NetworkData *netdat, double *inputNeurons);
int stochasticGradientDescent(NetworkInfo *netinf, NetworkData *netdat, double *learningRate, TrainingImage *training);
void freeWeights(NetworkInfo *netinf, NetworkData *netdat);
void freeBiases(NetworkInfo *netinf, NetworkData *netdat);
#endif
