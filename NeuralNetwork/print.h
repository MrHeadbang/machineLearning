#pragma once
#include "network.h"
#include "calc.h"

void printOutput(NetworkInfo *netinf, double *arr)
{
    int highestIndex = highestElementIndex(arr, netinf->neuronNum + netinf->layersNum - 1);
    
    printf("\n");
    for (uint8_t i = 0; i < *(netinf->neuronNum + netinf->layersNum - 1); i++)
    {
        printf("%d : %f", i, *(arr + i));
        if (highestIndex == i)
        {
            printf(" <-- Highest element");
        }
        printf("\n");
    }
    printf("\n");
}
