#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "training.h"
#include "print.h"
#include "files.h"

#define TRAINING_NUM 1000

int main(void)
{
    double learningRate = 0.005;

    srand((unsigned)time(NULL));
    uint8_t neuronNums[] = {4,14,30,14,4};
    NetworkInfo info = {5, neuronNums};

    NetworkData data = {initializeWeights(&info), initializeBiases(&info)};
    

    //printOutput(&info, forwardPass(&info, &data, trainingData->imageData));

    for (int j = 0; j < 1000; j++)
    {

        uint32_t correct = 0;
        TrainingImage *trainingData = generateTrainingImages(TRAINING_NUM);
        for (int i = 0; i < TRAINING_NUM; i++)
        {
            if (stochasticGradientDescent(&info, &data, &learningRate, trainingData + i))
            {
                correct++;
            }
        }
        free(trainingData);
        printf("%d : %f\n", j, 100 * (double)correct / ((double)TRAINING_NUM));
    }

    //printOutput(&info, (trainingData)->imageData);
    double a[] = {0.9,0.1,0.9,0.1};
    printOutput(&info, forwardPass(&info, &data, a));

    
    freeWeights(&info, &data);
    freeBiases(&info, &data);
    return EXIT_SUCCESS;
}
