// Neural Networks in C♯
// File name: MnistCnn.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.DataSources;
using NeuralNetworks.Layers;
using NeuralNetworks.LearningRates;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.Optimizers;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworksExamples;

class MnistConvModel(SeededRandom? random)
    : BaseModel<float[,,,], float[,]>(new SoftmaxCrossEntropyLoss(), random)
{
    protected override LayerListBuilder<float[,,,], float[,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);
        // ParamInitializer initializer = new RangeInitializer(1, 1);
        Dropout4D? dropout = new(0.85f, Random);

        return AddLayer(new Conv2DLayer(
                filters: 32, // 16,
                kernelSize: 3,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer,
                dropout: dropout
            ))
            .AddLayer(new FlattenLayer())
            .AddLayer(new DenseLayer(10, new Linear(), initializer));
    }

}

class MnistCnn
{
    const int RandomSeed = 251225;
    const int Epochs = 15;
    const int BatchSize = 100;
    const int EvalEveryEpochs = 3;
    const int LogEveryEpochs = 1;

    public static void Run()
    {
        ILogger<Trainer4D> logger = Program.LoggerFactory.CreateLogger<Trainer4D>();

        // rows - batch
        // cols - features
        float[,] train = LoadCsv("..\\..\\..\\..\\..\\data\\mnist\\mnist_train_small.csv");
        float[,] test = LoadCsv("..\\..\\..\\..\\..\\data\\mnist\\mnist_test.csv");

        (float[,,,] xTrain, float[,] yTrain) = Split(train);
        (float[,,,] xTest, float[,] yTest) = Split(test);

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

        SimpleDataSource<float[,,,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);
        SeededRandom commonRandom = new(RandomSeed);

        // Create a model

        MnistConvModel model = new(commonRandom);

        WriteLine("\nStart training...");

        LearningRate learningRate = new ExponentialDecayLearningRate(0.003f, 0.0004f);
        Trainer4D trainer = new(
            model,
            // new GradientDescentMomentumOptimizer(learningRate, 0.9f), 
            new AdamOptimizer(learningRate, 0.89f, 0.99f),
            random: commonRandom,
            logger: logger
        )
        {
            Memo = $"Class: {nameof(MnistCnn)}."
        };

        trainer.Fit(
            dataSource,
            s_evalFunction,
            epochs: Epochs,
            evalEveryEpochs: EvalEveryEpochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize
        );
    }

    private static readonly EvalFunction<float[,,,], float[,]> s_evalFunction = (model, xEvalTest, yEvalTest, predictionLogits) =>
    {
        float[,] prediction;
        if (predictionLogits != null)
        {
            prediction = predictionLogits;
        }
        else
        {
            prediction = model.Forward(xEvalTest, true);
        }

        // predictionArgmax is an array of predicted digits for each sample.
        int[] predictionArgmax = prediction.Argmax();

        int rows = predictionArgmax.GetLength(0);

        Debug.Assert(rows == yEvalTest.GetLength(0), "Number of samples in prediction and yEvalTest do not match.");

        int hits = 0;
        for (int row = 0; row < rows; row++)
        {
            int predictedDigit = predictionArgmax[row];

            // yEvalTest is a one-hot table.
            if (yEvalTest[row, predictedDigit] == 1f)
                hits++;
        }

        float accuracy = (float)hits / rows;
        return accuracy;
    };

    private static (float[,,,] xTest, float[,] yTest) Split(float[,] source)
    {
        // Split into xTest (all columns except the first one) and yTest (a one-hot table from the first column with values from 0 to 9).

        float[,] xTest2D = source.GetColumns(1..source.GetLength(1));
        float[,] yTest = source.GetColumn(0);

        Debug.Assert(xTest2D.GetLength(1) == 28 * 28);

        // Convert yTest to a one-hot table.
        int yTestRows = yTest.GetLength(0);
        float[,] oneHot = new float[yTestRows, 10];
        for (int row = 0; row < yTestRows; row++)
        {
            int value = Convert.ToInt32(yTest[row, 0]);
            oneHot[row, value] = 1f;
        }

        int xTestRows = xTest2D.GetLength(0);
        int xTestCols = xTest2D.GetLength(1);
        float[,,,] xTest4D = new float[xTestRows, 1, 28, 28];

        for (int row = 0; row < xTestRows; row++)
        {
            for (int col = 0; col < xTestCols; col++)
            {
                //int x = col % 28;
                //int y = col / 28;
                xTest4D[row, 0, col / 28, col % 28] = xTest2D[row, col];
            }
        }

        return (xTest4D, oneHot);
    }
}
