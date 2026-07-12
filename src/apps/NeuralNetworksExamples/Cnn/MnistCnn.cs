// Neural Networks in C♯
// File name: MnistCnn.cs
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
using static NeuralNetworksExamples.Utils;

namespace NeuralNetworksExamples.Cnn;

// For the current configuration and hyperparameters, the model achieves the accuracy:
// 97,83% - CpuSpansParallel
// 97,81% - Gpu
// 97,31% - CpuSpans, CpuArrays

internal class MnistConvModel(SeededRandom? random)
    : BaseModel<float[,,,], float[,]>(new SoftmaxCrossEntropyLoss(), random)
{
    protected override LayerListBuilder<float[,,,], float[,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);
        Dropout4D dropout = new(0.80f, Random);

        return
            AddLayer(new Conv2DLayer(
                kernels: 32, // 16,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new LeakyReLU4D(),
                paramInitializer: initializer,
                dropout: dropout
            ))
            .AddLayer(new FlattenLayer())
            .AddLayer(new DenseLayer(10, new Linear(), initializer));
    }

}

internal class MnistCnn
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
        ILogger logger = Program.LoggerFactory.CreateLogger<MnistCnn>();

        // rows - batch
        // cols - features
        float[,] train = GetMnistTrainData();
        (float[,] xTrain, float[,] yTrain) = SplitFeaturesAndEncodeLabels(train);
        float[,,,] xTrain4D = ReshapeTo4D(xTrain);

        float[,] test = GetMnistTestData();
        (float[,] xTest, float[,] yTest) = SplitFeaturesAndEncodeLabels(test);
        float[,,,] xTest4D = ReshapeTo4D(xTest);

        // Standardize data
        // We can standardize all features (columns) together because they are all in the same scale (pixel values from 0 to 255) and have similar meaning (brightness). We calculate mean and stdDev on the training set only, because in a real-world scenario we would not have access to the test set during training.
        // This means that before inference, we need to apply the same standardization to new data as we did to the training data.
        WriteLine("Standardize to mean 0 and variance 1 for all features together...");

        (float mean, float stdDev) = StandardizeInPlace(xTrain4D);
        ApplyStandardizationInPlace(xTest4D, mean, stdDev);

        SimpleDataSource<float[,,,], float[,]> dataSource = new(xTrain4D, yTrain, xTest4D, yTest);
        SeededRandom commonRandom = new(RandomSeed);

        // Create a model

        MnistConvModel model = new(commonRandom);

        WriteLine("\nStart training...");

        LearningRate learningRate = new ExponentialDecayLearningRate(InitialLearningRate, FinalLearningRate, 10);
        Trainer<float[,,,], float[,]> trainer = new(
            model,
            // new GradientDescentMomentumOptimizer(learningRate, 0.9f), 
            new AdamOptimizer(learningRate, AdamBeta1, AdamBeta2),
            random: commonRandom,
            logger: logger
        )
        {
            Memo = $"Calling class: {nameof(MnistCnn)}. BiasAdd array"
        };

        trainer.Fit(
            dataSource,
            s_evalFunction,
            epochs: Epochs,
            evalEveryEpochs: EvalEveryEpochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize,
            saveParamsOnBestLoss: false
        );

        // Display some examples of predictions vs actual values for the test set for the digit "3", which is the most difficult digit to classify in the MNIST dataset (as Copilot says, I don't know if it's true 😉)

        float[,] logits = model.Forward(xTest4D, true);
        DisplayDigit3PredictionExamples(yTest, logits, xTest, "cnn");
    }

    private static readonly EvalFunction<float[,,,], float[,]> s_evalFunction = (model, xEvalTest, yEvalTest, predictionLogits) =>
    {
        float[,] prediction = predictionLogits ?? model.Forward(xEvalTest, true);

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

}