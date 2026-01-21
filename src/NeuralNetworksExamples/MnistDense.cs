// Neural Networks in C♯
// File name: MnistDense.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Linq;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.DataSources;
using NeuralNetworks.Layers;
using NeuralNetworks.LearningRates;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.Operations.Dropouts;
using NeuralNetworks.Optimizers;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworksExamples;

// For the current configuration and hyperparameters, the model achieves the accuracy:
// 97,07% - CpuSpansParallel
// 97,05% - Gpu
// 97,07% - CpuSpans, CpuArrays

internal class MnistDenseModel: BaseModel<float[,], float[,]>
{
    public MnistDenseModel(SeededRandom? random) : base(new SoftmaxCrossEntropyLoss(), random)
    {
    }

    public MnistDenseModel(string? modelFilePath) : base(new SoftmaxCrossEntropyLoss(), null, modelFilePath)
    {
    }

    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        GlorotInitializer initializer = new(Random);

        return
             AddLayer(new DenseLayer(178, new LeakyReLU2D(), initializer, new Dropout2D(0.8f, Random)))
            .AddLayer(new DenseLayer(46, new LeakyReLU2D(), initializer, new Dropout2D(0.8f, Random)))
            .AddLayer(new DenseLayer(10, new Linear(), initializer));
    }
}

internal class MnistDense
{
    private const int RandomSeed = 44; // From Mickiewicz's poetry.
    private const int Epochs = 15;
    private const int BatchSize = 400;
    private const int EvalEveryEpochs = 2;
    private const int LogEveryEpochs = 1;

    private const float InitialLearningRate = 0.002f;
    private const float FinalLearningRate = 0.0005f;
    private const float AdamBeta1 = 0.89f;
    private const float AdamBeta2 = 0.99f;

    public static void Run()
    {
        ILogger<Trainer2D> logger = Program.LoggerFactory.CreateLogger<Trainer2D>();

        // rows - batch
        // cols - features
        float[,] train = LoadCsv("..\\..\\..\\..\\..\\data\\MNIST\\mnist_train_small.csv");
        float[,] test = LoadCsv("..\\..\\..\\..\\..\\data\\MNIST\\mnist_test.csv");

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

        MnistDenseModel model = new(commonRandom);

        WriteLine("\nStart training...");

        LearningRate learningRate = new ExponentialDecayLearningRate(InitialLearningRate, FinalLearningRate);
        Trainer2D trainer = new(
            model,
            // new GradientDescentMomentumOptimizer(learningRate, 0.9f), 
            new AdamOptimizer(learningRate, beta1: AdamBeta1, beta2: AdamBeta2),
            random: commonRandom,
            logger: logger
        )
        {
            Memo = $"Calling class: {nameof(MnistDense)}."
        };

        trainer.Fit(
            dataSource,
            s_evalFunction,
            epochs: Epochs,
            evalEveryEpochs: EvalEveryEpochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize,
            displayDescriptionOnStart: true
        );

        // Save the model
        string modelPath = "MnistDenseModel.json";
        model.SaveParams(modelPath, "Final trained model.");
    }

    private static readonly EvalFunction<float[,], float[,]> s_evalFunction = (model, xEvalTest, yEvalTest, predictionLogits) =>
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

    internal static void LoadAndEvaluate()
    {
        // Load test data
        float[,] test = LoadCsv("..\\..\\..\\..\\..\\data\\MNIST\\mnist_test.csv");
        (float[,] xTest, float[,] yTest) = Split(test);

        // Standardize data
        float mean = xTest.Mean();
        xTest.AddInPlace(-mean);
        float stdDev = xTest.StdDev();
        xTest.DivideInPlace(stdDev);

        // Load the model
        string modelPath = "MnistDenseModel.json";
        MnistDenseModel model = new(modelPath);
        //model.LoadParams(modelPath);

        // Evaluate
        float[,] prediction = model.Forward(xTest, true);

        // predictionArgmax is an array of predicted digits for each sample.
        int[] predictionArgmax = prediction.Argmax();
        int rows = predictionArgmax.GetLength(0);

        Debug.Assert(rows == yTest.GetLength(0), "Number of samples in prediction and yEvalTest do not match.");

        int hits = 0;
        for (int row = 0; row < rows; row++)
        {
            int predictedDigit = predictionArgmax[row];
            if (yTest[row, predictedDigit] == 1f)
                hits++;
        }
        float accuracy = (float)hits / rows;
        WriteLine($"Model accuracy on test set: {accuracy:P2}");
    }
}
