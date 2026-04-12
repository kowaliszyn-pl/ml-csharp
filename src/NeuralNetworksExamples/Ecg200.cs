// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

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

/*
Train loss (average): 0,16014086
Test loss: 0,3037759
Train eval: 96,00%
Test eval: 89,00%
*/

internal class Ecg200Model(SeededRandom? random)
    : BaseModel<float[,,], float[,]>(new BinaryCrossEntropyLoss(), random)
{
    protected override LayerListBuilder<float[,,], float[,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);
        Dropout3D dropout = new(0.76f, Random);
        return
            AddLayer(new Conv1DLayer(
                kernels: 16,
                kernelLength: 5,
                stride: 1,
                activationFunction: new ReLU3D(),
                paramInitializer: initializer,
                dropout: dropout
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
    private const int RandomSeed = 260221;
    private const int Epochs = 340;
    private const int BatchSize = 25;
    private const int EvalEveryEpochs = 20;
    private const int LogEveryEpochs = 10;

    private const float InitialLearningRate = 0.009f;
    private const float FinalLearningRate = 0.00165f;
    private const float AdamBeta1 = 0.89f;
    private const float AdamBeta2 = 0.99f;

    internal static void Run()
    {
        ILogger logger = Program.LoggerFactory.CreateLogger<Ecg200>();

        // rows - batch
        // cols - features
        float[,] train = LoadTsv("..\\..\\..\\..\\..\\data\\ecg200\\ECG200_TRAIN.tsv");
        float[,] test = LoadTsv("..\\..\\..\\..\\..\\data\\ecg200\\ECG200_TEST.tsv");

        // float[,] y = [batch, 0] = class 1 probability

        (float[,,] xTrain, float[,] yTrain, _) = Split(train);
        (float[,,] xTest, float[,] yTest, float[,] testImagesForDrawing) = Split(test);

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

        // Create a model

        Ecg200Model model = new(commonRandom);

        WriteLine("\nStart training...");

        LearningRate learningRate = new ExponentialDecayLearningRate(InitialLearningRate, FinalLearningRate, 0);
        Trainer<float[,,], float[,]> trainer = new(
            model,
            new AdamOptimizer(learningRate, AdamBeta1, AdamBeta2),
            random: commonRandom,
            logger: logger
        )
        {
            Memo = $"Calling class: {nameof(Ecg200)}"
        };

        trainer.Fit(
            dataSource,
            s_evalFunction,
            epochs: Epochs,
            evalEveryEpochs: EvalEveryEpochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize,
            showTrainEval: true
        );

        // Now let's display some examples of predictions vs actual values for the test set.

        float[,] predictions = model.Forward(xTest, true);
        Utils.DisplayClassificationPredictionExamples(yTest, predictions, testImagesForDrawing, "cnn");
    }

    private static readonly EvalFunction<float[,,], float[,]> s_evalFunction = (model, xEvalTest, yEvalTest, predictionLogits) =>
    {
        float[,] prediction; // [batch, 0] = probability of being normal (class 1)
        if (predictionLogits != null)
        {
            prediction = predictionLogits;
        }
        else
        {
            prediction = model.Forward(xEvalTest, true);
        }

        int rows = prediction.GetLength(0);

        Debug.Assert(rows == yEvalTest.GetLength(0), "Number of samples in prediction and yEvalTest do not match.");

        int hits = 0;
        for (int row = 0; row < rows; row++)
        {
            // yEvalTest is [batch, 0] = 100%, if normal (class 1), or 0%, if abnormal (class 0). So we check if the predicted class (normal or abnormal) matches the actual class.
            
            float predictedProbabilityOfNormal = prediction[row, 0];
            float actualClass = yEvalTest[row, 0]; // 1 for normal, 0 for abnormal
            bool actualNormalClass = actualClass == 1f;

            if ((predictedProbabilityOfNormal >= 0.5f && actualNormalClass) 
                || (predictedProbabilityOfNormal < 0.5f && !actualNormalClass))
                hits++;
        }

        float accuracy = (float)hits / rows;
        return accuracy;
    };

    private static (float[,,] xData, float[,] yData, float[,] xData2D) Split(float[,] source)
    {
        // Split into xData (all columns except the first one) and yData (the first column with values 1 or -1, where 1 means normal and -1 means abnormal (myocardial infarction)).

        float[,] xData2D = source.GetColumns(1..source.GetLength(1));
        float[,] yData = source.GetColumn(0);

        for (int row = 0; row < yData.GetLength(0); row++)
        {
            Debug.Assert(yData[row, 0] == 1f || yData[row, 0] == -1f, $"Expected values in the first column to be either 1 or -1, but got {yData[row, 0]} at row {row}.");
            // Convert the values in yData to 100% for normal (1) and 0% for abnormal (-1).
            yData[row, 0] = yData[row, 0] == 1f ? 1f : 0f;
        }

        int xDataRows = xData2D.GetLength(0);
        int xDataCols = xData2D.GetLength(1);
        float[,,] xData = new float[xDataRows, 1, xDataCols];
        for (int row = 0; row < xDataRows; row++)
        {
            for (int col = 0; col < xDataCols; col++)
            {
                xData[row, 0 /* one input channel */, col] = xData2D[row, col];
            }
        }

        Debug.Assert(xData.GetLength(0) == yData.GetLength(0), "Number of samples in xData and yData do not match.");
        return (xData, yData, xData2D);
    }
}