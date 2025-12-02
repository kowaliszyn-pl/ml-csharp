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
using NeuralNetworks.Operations;
using NeuralNetworks.Optimizers;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworksExamples;

class BostonHousingModel(SeededRandom? random)
    : Model<float[,], float[,]>(new MeanSquaredError(), random)
{
    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        GlorotInitializer initializer = new(Random);
        Dropout2D? dropout1 = new(0.85f, Random);
        Dropout2D? dropout2 = new(0.85f, Random);

        return AddLayer(new DenseLayer(4, new Sigmoid(), initializer, dropout1))
            .AddLayer(new DenseLayer(1, new Linear(), initializer));
    }

}

class BostonHousing
{
    const int RandomSeed = 251203;
    const float TestSplitRatio = 0.7f;
    const int Epochs = 48_000;
    const int BatchSize = 400;
    const int EvalEveryEpochs = 2_000;
    const int LogEveryEpochs = 2_000;

    public static void Run()
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

        BostonHousingModel model = new(commonRandom);

        WriteLine("\nStart training...\n");

        LearningRate learningRate = new ExponentialDecayLearningRate(0.19f, 0.05f);
        Trainer2D trainer = new(model, new StochasticGradientDescentMomentum(learningRate, 0.9f), random: commonRandom, logger: logger)
        {
            Memo = $"Class: {nameof(Mnist)}."
        };

        trainer.Fit(
            dataSource,
            epochs: Epochs,
            evalEveryEpochs: EvalEveryEpochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize
        );
    }

    static (float[,] TrainData, float[,] TestData) GetData()
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
