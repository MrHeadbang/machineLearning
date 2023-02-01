#include <stdio.h>
#include <stdlib.h>
#include "calc.h"
#include "network.h"


double ***initializeWeights(NetworkInfo *netinf)
{
    double ***networkWeights = (double***)malloc((netinf->layersNum) * sizeof(double**));

    //Iterate over every layer except the input layer
    for (uint8_t i = 1; i < netinf->layersNum; i++)
    {
        double **layerWeights = (double**)malloc(*((netinf->neuronNum) + i) * sizeof(double*));
        //Iterate over every neuron
        for (uint8_t j = 0; j < *((netinf->neuronNum) + i); j++)
        {
            double *neuronWeights = (double*)malloc(*((netinf->neuronNum) + i - 1) * sizeof(double));

            //Iterate over previous neuron layer and set default weight
            for (uint8_t k = 0; k < *((netinf->neuronNum) + i - 1); k++)
            {
                *(neuronWeights + k) = randomDouble(0, 1.0); //Any first value of each weight
            }
            *(layerWeights + j) = neuronWeights;
        }
        *(networkWeights + i - 1) = layerWeights;
    }
    
    return networkWeights;
}

void freeWeights(NetworkInfo *netinf, NetworkData *netdat)
{
    double ***weights = netdat->weights;
    for (uint8_t i = 0; i < netinf->layersNum - 1; i++)
    {
        for (uint8_t j = 0; j < *((netinf->neuronNum) + 1 + i); j++)
        {
            free(*(*(weights + i) + j));
        }
        free(*(weights + i));
    }
    free(weights);
}

double **initializeBiases(NetworkInfo *netinf)
{
    double **networkBiases = (double**)malloc((netinf->layersNum) * sizeof(double*));

    //Iterate over every layer except the input layer
    for (uint8_t i = 1; i < netinf->layersNum; i++)
    {
        double *layerBiases = (double*)malloc(*((netinf->neuronNum) + i) * sizeof(double));

        //Iterate over every neuron and set default bias
        for (uint8_t j = 0; j < *((netinf->neuronNum) + i); j++)
        {
            *(layerBiases + j) = randomDouble(0, 1.0);
        }
        *(networkBiases + i - 1) = layerBiases;
    }
    return networkBiases;
}

void freeBiases(NetworkInfo *netinf, NetworkData *netdat)
{
    double **biases = netdat->biases;
    for (uint8_t i = 0; i < netinf->layersNum - 1; i++)
    {
        free(*(biases + i));
    }
    free(biases);
}

double *calculateLayerNeurons(NetworkInfo *netinf, uint8_t netinfIndex, double *inputNeurons, double **layerWeights, double *layerBiases)
{
    //Allocate memory in size of the next layer
    double *outputNeurons = (double*)malloc(*(netinf->neuronNum + netinfIndex + 1) * sizeof(double));

    //Iterate over current layer
    for (uint8_t i = 0; i < *(netinf->neuronNum + netinfIndex + 1); i++)
    {
        double *currentWeights = *(layerWeights + i);

        double outputNeuron = *(layerBiases + i);

        for (uint8_t j = 0; j < *(netinf->neuronNum + netinfIndex); j++)
        {
            outputNeuron += *(currentWeights + j) * *(inputNeurons + j);
        }
        *(outputNeurons + i) = Sigmoid(outputNeuron);
    }
    return outputNeurons;
}

double *forwardPass(NetworkInfo *netinf, NetworkData *netdat, double *inputNeurons)
{
    //Iterate over layers except the last one
    for (uint8_t i = 0; i < netinf->layersNum - 1; i++)
    {
        //Storing the weight and bias of the current layer
        double **currentLayerWeights = *(netdat->weights + i);
        double *currentLayerBiases = *(netdat->biases + i);

        //Calculate the new neurons using the previous values
        inputNeurons = calculateLayerNeurons(netinf, i, inputNeurons, currentLayerWeights, currentLayerBiases);
    }
    return inputNeurons;
}

int stochasticGradientDescent(NetworkInfo *netinf, NetworkData *netdat, double *learningRate, TrainingImage *training)
{
    double **neuronValues = (double**)malloc(netinf->layersNum * sizeof(double*));

    double *inputNeurons = training->imageData;
    uint8_t inputIndex = training->imageIndex;

    *neuronValues = inputNeurons;

    for (uint8_t i = 0; i < netinf->layersNum - 1; i++)
    {
         //Storing the weight and bias of the current layer
        double **currentLayerWeights = *(netdat->weights + i);
        double *currentLayerBiases = *(netdat->biases + i);

        //Calculate the new neurons using the previous values
        inputNeurons = calculateLayerNeurons(netinf, i, inputNeurons, currentLayerWeights, currentLayerBiases);
        *(neuronValues + i + 1) = inputNeurons;
    }

    //Values of output neurons
    double *outputNeurons = *(neuronValues + netinf->layersNum - 1);

    //Index of highest element in output
    uint8_t highestIndex = highestElementIndex(outputNeurons, netinf->neuronNum + netinf->layersNum - 1);

    double *perfectResult = expectedOutput(&inputIndex, netinf->neuronNum + netinf->layersNum - 1);

    //2D Array of all deltas for every layer starting from the back
    double **neuronDeltas = (double**)malloc(netinf->layersNum * sizeof(double*));

    double *lastDeltas = (double*)malloc(*(netinf->neuronNum + netinf->layersNum - 1) * sizeof(double));
    
    //Calculate deltas of the last layer
    for (uint8_t i = 0; i < *(netinf->neuronNum + netinf->layersNum - 1); i++)
    {
        *(lastDeltas + i) = *(perfectResult + i) - *(outputNeurons + i);
    }
    *neuronDeltas = lastDeltas;

    //Calculate delta of every layer
    for (int8_t i = netinf->layersNum - 2; i >= 0; i--)
    {
        double **layerWeights = *(netdat->weights + i);
        double *previousDeltas = *(neuronDeltas + netinf->layersNum - 2 - i);
        double *layerDeltas = (double*)malloc(*(netinf->neuronNum + i) * sizeof(double));
        for (uint8_t j = 0; j < *(netinf->neuronNum + i); j++)
        {
            double deltaSum = 0.0;

            for (uint8_t k = 0; k < *(netinf->neuronNum + i + 1); k++)
            {
                deltaSum += *(*(layerWeights + k) + j) * *(previousDeltas + k);
            }
            *(layerDeltas + j) = deltaSum;
        }
        *(neuronDeltas + netinf->layersNum - i - 1) = layerDeltas;
    }

    //Change the weights using delta and the neuronvalues for every layer
    for (uint8_t i = 0; i < netinf->layersNum - 1; i++)
    {
        double *currentDeltas = *(neuronDeltas + netinf->layersNum - i - 2);
        double *currentNeurons = *(neuronValues + i);
        for (uint8_t deltaIterator = 0; deltaIterator < *(netinf->neuronNum + i + 1); deltaIterator++)
        {
            for (uint8_t neuronIterator = 0; neuronIterator < *(netinf->neuronNum + i); neuronIterator++)
            {
                *(*(*(netdat->weights + i) + deltaIterator) + neuronIterator) += *learningRate * *(currentDeltas + deltaIterator) * *(currentNeurons + neuronIterator);
            }
            *(*(netdat->biases + i) + deltaIterator) -= *learningRate * *(currentDeltas + deltaIterator);
        }

    } 
    free(neuronDeltas);
    free(lastDeltas);
    free(neuronValues);
    if (highestIndex == inputIndex)
    {
        return 1;
    }
    return 0;
}
