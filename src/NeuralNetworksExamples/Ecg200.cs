// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworksExamples;

internal class Ecg200Model(SeededRandom? random)
    : BaseModel<float[,,], float[,]>(new SoftmaxCrossEntropyLoss(), random)
{
    protected override LayerListBuilder<float[,,], float[,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);
        return
            AddLayer(new Conv1DLayer(
                kernels: 16,
                kernelLength: 5,
                stride: 1,
                activationFunction: new ReLU3D(),
                paramInitializer: initializer
            ))
            .AddLayer(new MaxPooling3DLayer(2))
            .AddLayer(new Conv1DLayer(
                kernels: 32,
                kernelLength: 3,
                stride: 1,
                activationFunction: new ReLU3D(),
                paramInitializer: initializer
            ))
            .AddLayer(new GlobalAveragePooling3DLayer())
            .AddLayer(new DenseLayer(1, new Sigmoid(), initializer));
    }
}

internal class Ecg200
{
    internal static void Run()
    {
        ILogger<Trainer4D> logger = Program.LoggerFactory.CreateLogger<Trainer4D>();

        // rows - batch
        // cols - features
        float[,] train = LoadTsv("..\\..\\..\\..\\..\\data\\ecg200\\ECG200_TRAIN.tsv");
        float[,] test = LoadTsv("..\\..\\..\\..\\..\\data\\ecg200\\ECG200_TEST.tsv");

        (float[,] xTrain, bool[,] yTrain) = Split(train);
        (float[,] xTest, bool[,] yTest) = Split(test);
    }

    private static (float[,] xTest, bool[,] yTest) Split(float[,] source)
    {
        // Split into xTest (all columns except the first one) and yTest (the first column with values 1 or -1, where 1 means normal and -1 means abnormal (myocardial infarction)).

        float[,] xTest = source.GetColumns(1..source.GetLength(1));
        float[,] yTestAsNumber = source.GetColumn(0);

        bool[,] yTest = new bool[yTestAsNumber.GetLength(0), 1];
        for (int row = 0; row < yTestAsNumber.GetLength(0); row++)
        {
            Debug.Assert(yTestAsNumber[row, 0] == 1f || yTestAsNumber[row, 0] == -1f, $"Expected values in the first column to be either 1 or -1, but got {yTestAsNumber[row, 0]} at row {row}.");
            yTest[row, 0] = yTestAsNumber[row, 0] == 1f; // 1 = true = normal, -1 = false = abnormal (myocardial infarction)
        }

        Debug.Assert(xTest.GetLength(0) == yTest.GetLength(0), "Number of samples in xTest and yTest do not match.");

        return (xTest, yTest);
    }
}