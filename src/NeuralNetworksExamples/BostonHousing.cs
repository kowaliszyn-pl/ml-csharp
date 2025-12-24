// Neural Networks in C♯
// File name: BostonHousing.cs
// www.kowaliszyn.pl, 2025

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.DataSources;
using NeuralNetworks.Layers;
using NeuralNetworks.LearningRates;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.Optimizers;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworksExamples;

class BostonHousingModel(SeededRandom? random)
    : BaseModel<float[,], float[,]>(new MeanSquaredError(), random)
{
    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        GlorotInitializer initializer = new(Random);

        return AddLayer(new DenseLayer(4, new Tanh2D(), initializer))
            .AddLayer(new DenseLayer(1, new Linear(), initializer));
    }
}

internal class BostonHousing
{
    private const int RandomSeed = 251113;
    private const float TestSplitRatio = 0.7f;
    private const int Epochs = 48_000;
    private const int BatchSize = 400;
    private const int EvalEveryEpochs = 2_000;
    private const int LogEveryEpochs = 2_000;

    public static void Run(bool useCustomModel)
    {
        ILogger<Trainer2D> logger = Program.LoggerFactory.CreateLogger<Trainer2D>();

        // Get data

        (float[,] trainData, float[,] testData) = GetData();

        // Copy trainData and testData to XTrain, YTrain, XTest, YTest

        int inputFeatureCount = trainData.GetLength(1) - 1;
        int nTrain = trainData.GetLength(0);
        int nTest = testData.GetLength(0);

        float[,] XTrain = new float[nTrain, inputFeatureCount];
        float[,] YTrain = new float[nTrain, 1];

        float[,] XTest = new float[nTest, inputFeatureCount];
        float[,] YTest = new float[nTest, 1];

        // Prepare feature matrix XTrain and target vector YTrain
        for (int i = 0; i < nTrain; i++)
        {
            for (int j = 0; j < inputFeatureCount; j++)
            {
                XTrain[i, j] = trainData[i, j];
            }

            // Target values
            YTrain[i, 0] = trainData[i, inputFeatureCount];
        }

        // Prepare feature matrix XTest and target vector YTest
        for (int i = 0; i < nTest; i++)
        {
            for (int j = 0; j < inputFeatureCount; j++)
            {
                XTest[i, j] = testData[i, j];
            }
            // Target values
            YTest[i, 0] = testData[i, inputFeatureCount];
        }

        SimpleDataSource<float[,], float[,]> dataSource = new(XTrain, YTrain, XTest, YTest);
        SeededRandom commonRandom = new(RandomSeed);

        // Build a model

        Model<float[,], float[,]> model;
        if (useCustomModel)
        {
            model = new BostonHousingModel(commonRandom);
        }
        else
        {
            model = new GenericModel<float[,], float[,]>(

                layerListBuilder: LayerListBuilder<float[,], float[,]>
                    .AddLayer(new DenseLayer(neurons: 4, new Tanh2D(), new GlorotInitializer(commonRandom)))
                    .AddLayer(new DenseLayer(neurons: 1, new Linear(), new GlorotInitializer(commonRandom))),

                lossFunction: new MeanSquaredError(),
                random: commonRandom);
        }

        ExponentialDecayLearningRate learningRate = new(
            initialLearningRate: 0.0009f,
            finalLearningRate: 0.0005f
        );

        Trainer2D trainer = new(
            model,
            new GradientDescentMomentumOptimizer(learningRate, 0.9f),
            // new AdamOptimizer(learningRate),
            random: commonRandom,
            logger: logger)
        {
            Memo = $"Class: {nameof(BostonHousing)}."
        };

        trainer.Fit(
            dataSource,
            epochs: Epochs,
            evalEveryEpochs: EvalEveryEpochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize
        );

        WriteLine();
        WriteLine("Sample predictions vs actual values:");
        WriteLine();
        WriteLine($"{"Sample No",14}{"Predicted",14}{"Actual",14}");
        WriteLine();

        // Show predictions for the test set

        int[] showTestSamples = { 0, 1, 2, nTest - 3, nTest - 2, nTest - 1 };

        // Do a forward pass for all test samples at once

        float[,] predictions = model.Forward(XTest, true);
        foreach (int sampleIndex in showTestSamples)
        {
            float predictedValue = predictions[sampleIndex, 0];
            float actualValue = YTest[sampleIndex, 0];
            WriteLine($"{sampleIndex + 1,14}{predictedValue,14:F4}{actualValue,14:F4}");
        }

    }

    private static (float[,] TrainData, float[,] TestData) GetData()
    {
        float[,] bostonData = LoadCsv("..\\..\\..\\..\\..\\data\\Boston\\BostonHousing.csv", 1);

        // Number of independent variables
        int inputFeatureCount = bostonData.GetLength(1) - 1;

        // Standardize each feature column (mean = 0, stddev = 1) except the target variable (last column)
        // Note: the upper bound in Range is exclusive, so we use inputFeatureCount to exclude the last column
        bostonData.Standardize(0..inputFeatureCount);

        // Permute the data randomly
        bostonData.PermuteInPlace(RandomSeed);

        // Return train and test data split by ratio
        return bostonData.SplitRowsByRatio(TestSplitRatio);
    }
}
