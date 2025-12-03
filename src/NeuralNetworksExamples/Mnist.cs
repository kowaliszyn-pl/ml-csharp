// Neural Networks in C♯
// File name: Mnist.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.DataSources;
using NeuralNetworks.Layers;
using NeuralNetworks.LearningRates;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Operations;
using NeuralNetworks.Optimizers;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworksExamples;

class MnistModel(SeededRandom? random)
    : Model<float[,], float[,]>(new SoftmaxCrossEntropyLoss(), random)
{
    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        GlorotInitializer initializer = new(Random);
        Dropout2D? dropout1 = new(0.85f, Random);
        Dropout2D? dropout2 = new(0.85f, Random);

        return AddLayer(new DenseLayer(178, new Tanh2D(), initializer, dropout1))
            .AddLayer(new DenseLayer(46, new Tanh2D(), initializer, dropout2))
            .AddLayer(new DenseLayer(10, new Linear(), initializer));
    }
}

class Mnist
{
    const int RandomSeed = 251203;
    const int Epochs = 10;
    const int BatchSize = 100;
    const int EvalEveryEpochs = 2;
    const int LogEveryEpochs = 1;

    public static void Run()
    {
        ILogger<Trainer2D> logger = Program.LoggerFactory.CreateLogger<Trainer2D>();

        // rows - batch
        // cols - features
        float[,] train = LoadCsv("..\\..\\..\\..\\..\\data\\mnist\\mnist_train_small.csv");
        float[,] test = LoadCsv("..\\..\\..\\..\\..\\data\\mnist\\mnist_test.csv");

        (float[,] xTrain, float[,] yTrain) = Split(train);
        (float[,] xTest, float[,] yTest) = Split(test);

        // Standardize data
        // We can standardize all features (columns) together because they are all in the same scale (pixel values from 0 to 255) and have similar meaning (brightness). We calculate mean and stdDev on the training set only, because in a real-world scenario we would not have access to the test set during training.
        // This means that before inference, we need to apply the same standardization to new data as we did to the training data.
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

        SimpleDataSource<float[,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);
        SeededRandom commonRandom = new(RandomSeed);

        // Create a model

        MnistModel model = new(commonRandom);

        WriteLine("\nStart training...");

        LearningRate learningRate = new ExponentialDecayLearningRate(0.19f, 0.05f);
        Trainer2D trainer = new(model, new StochasticGradientDescentMomentum(learningRate, 0.9f), random: commonRandom, logger: logger)
        {
            Memo = $"Class: {nameof(Mnist)}."
        };

        trainer.Fit(
            dataSource,
            EvalFunction,
            epochs: Epochs,
            evalEveryEpochs: EvalEveryEpochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize
        );
    }

    private static float EvalFunction(Model<float[,], float[,]> neuralNetwork, float[,] xEvalTest, float[,] yEvalTest)
    {
        // 'prediction' is a one-hot table with the predicted digit.
        float[,] prediction = neuralNetwork.Forward(xEvalTest, true);
        int[] predictionArgmax = prediction.Argmax();

        int rows = predictionArgmax.GetLength(0);

        Debug.Assert(rows == yEvalTest.GetLength(0), "Number of samples in prediction and yEvalTest do not match.");

        int hits = 0;
        for (int row = 0; row < rows; row++)
        {
            int predictedDigit = predictionArgmax[row];
            if (yEvalTest[row, predictedDigit] == 1f)
                hits++;
        }

        float accuracy = (float)hits / rows;
        return accuracy;
    }

    private static (float[,] xTest, float[,] yTest) Split(float[,] source)
    {
        // Split into xTest (all columns except the first one) and yTest (a one-hot table from the first column with values from 0 to 9).

        float[,] xTest = source.GetColumns(1..source.GetLength(1));
        float[,] yTest = source.GetColumn(0);

        // Convert yTest to a one-hot table.
        float[,] oneHot = new float[yTest.GetLength(0), 10];
        for (int row = 0; row < yTest.GetLength(0); row++)
        {
            int value = Convert.ToInt32(yTest[row, 0]);
            oneHot[row, value] = 1f;
        }

        return (xTest, oneHot);
    }
}
