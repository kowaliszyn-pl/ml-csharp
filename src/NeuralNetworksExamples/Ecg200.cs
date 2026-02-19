// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.DataSources;
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
    : BaseModel<float[,,], float[,]>(new BinaryCrossEntropyLoss(), random)
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
            .AddLayer(new MaxPooling1DLayer(2))
            .AddLayer(new Conv1DLayer(
                kernels: 32,
                kernelLength: 3,
                stride: 1,
                activationFunction: new ReLU3D(),
                paramInitializer: initializer
            ))
            .AddLayer(new GlobalAveragePooling1DLayer())

            // Probability of being normal (class 1)
            .AddLayer(new DenseLayer(1, new Sigmoid(), initializer));
    }
}

internal class Ecg200
{
    private const int RandomSeed = 44; // From Mickiewicz's poetry.
    private const int Epochs = 3; // 15;
    private const int BatchSize = 400;
    private const int EvalEveryEpochs = 2;
    private const int LogEveryEpochs = 1;

    private const float InitialLearningRate = 0.002f;
    private const float FinalLearningRate = 0.0005f;
    private const float AdamBeta1 = 0.89f;
    private const float AdamBeta2 = 0.99f;

    internal static void Run()
    {
        ILogger<Trainer4D> logger = Program.LoggerFactory.CreateLogger<Trainer4D>();

        // rows - batch
        // cols - features
        float[,] train = LoadTsv("..\\..\\..\\..\\..\\data\\ecg200\\ECG200_TRAIN.tsv");
        float[,] test = LoadTsv("..\\..\\..\\..\\..\\data\\ecg200\\ECG200_TEST.tsv");

        // float[,] y = [batch, class (1 or -1)]

        (float[,,] xTrain, float[,] yTrain) = Split(train);
        (float[,,] xTest, float[,] yTest) = Split(test);

        WriteLine("Standardize to mean 0 and variance 1 for all features together...");

        float mean = xTrain.Mean();
        WriteLine($"Current mean: {mean}. Scale data to mean 0...");
        xTrain.AddInPlace(-mean);
        xTest.AddInPlace(-mean);

        WriteLine($"xTrain min: {xTrain.Min()}");
        WriteLine($"xTest min: {xTest.Min()}");
        WriteLine($"xTrain max: {xTrain.Max()}");
        WriteLine($"xTest max: {xTest.Max()}");

        float stdDev = xTrain.StdDev();
        WriteLine($"\nCurrent stdDev: {stdDev}. Scale data to variance 1...");
        xTrain.DivideInPlace(stdDev);
        xTest.DivideInPlace(stdDev);
        WriteLine($"xTrain min: {xTrain.Min()}");
        WriteLine($"xTest min: {xTest.Min()}");
        WriteLine($"xTrain max: {xTrain.Max()}");
        WriteLine($"xTest max: {xTest.Max()}");

        SimpleDataSource<float[,,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);
        SeededRandom commonRandom = new(RandomSeed);
    }

    private static (float[,,] xTest, float[,] yTest) Split(float[,] source)
    {
        // Split into xTest (all columns except the first one) and yTest (the first column with values 1 or -1, where 1 means normal and -1 means abnormal (myocardial infarction)).

        float[,] xTest2D = source.GetColumns(1..source.GetLength(1));
        float[,] yTest = source.GetColumn(0);

        for (int row = 0; row < yTest.GetLength(0); row++)
        {
            Debug.Assert(yTest[row, 0] == 1f || yTest[row, 0] == -1f, $"Expected values in the first column to be either 1 or -1, but got {yTest[row, 0]} at row {row}.");

            // Convert the values in yTest to 100% for normal (1) and 0% for abnormal (-1).
            yTest[row, 0] = yTest[row, 0] == 1f ? 1f : 0f;
        }

        float[,,] xTest = new float[xTest2D.GetLength(0), 1, xTest2D.GetLength(1)];
        for (int row = 0; row < xTest2D.GetLength(0); row++)
        {
            for (int col = 0; col < xTest2D.GetLength(1); col++)
            {
                xTest[row, 0 /* one input channel */, col] = xTest2D[row, col];
            }
        }

        Debug.Assert(xTest.GetLength(0) == yTest.GetLength(0), "Number of samples in xTest and yTest do not match.");

        return (xTest, yTest);
    }
}