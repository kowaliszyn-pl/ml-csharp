// Neural Networks in C♯
// File name: SineFunction.cs
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
using NeuralNetworks.Optimizers;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;

using static System.Console;

namespace NeuralNetworksExamples;

/*
Train loss (average): 8,890273E-05
Test loss: 7,955757E-05
*/

internal class SineFunctionModel(SeededRandom? random)
    : BaseModel<float[,], float[,]>(new MeanSquaredErrorLoss(), random)
{
    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        GlorotInitializer initializer = new(Random);

        return AddLayer(new DenseLayer(32, new Tanh2D(), initializer))
            .AddLayer(new DenseLayer(32, new Tanh2D(), initializer))
            .AddLayer(new DenseLayer(1, new Linear(), initializer));
    }

}

internal class SineFunction
{
    private const int RandomSeed = 260221;

    public static void Run()
    {
        // Create data set
        int sampleCount = 1_000;
        List<(float x, float y)> data = [];

        for (int i = 0; i < sampleCount; i++)
        {
            float x = -MathF.PI + 2 * MathF.PI * i / sampleCount;
            float y = MathF.Sin(x);
            data.Add((x, y));
        }

        // Shuffle
        SeededRandom random = new(RandomSeed);
        data = [.. data.OrderBy(_ => random.Next())];

        // Split 80/20
        int trainSize = (int)(0.8f * sampleCount);
        int testSize = sampleCount - trainSize;

        List<(float x, float y)> train = data[..trainSize];
        List<(float x, float y)> test = data[trainSize..];

        Debug.Assert(train.Count + test.Count == sampleCount);

        // Create data source (float[] xTrain, float[] yTrain, float[] xTest, float[] yTest)
        float[,] xTrain = new float[trainSize, 1];
        float[,] yTrain = new float[trainSize, 1];
        for (int i = 0; i < trainSize; i++)
        {
            xTrain[i, 0] = train[i].x;
            yTrain[i, 0] = train[i].y;
        }

        float[,] xTest = new float[testSize, 1];
        float[,] yTest = new float[testSize, 1];
        for (int i = 0; i < testSize; i++)
        {
            xTest[i, 0] = test[i].x;
            yTest[i, 0] = test[i].y;
        }

        SimpleDataSource<float[,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);

        WriteLine("Standardize to mean 0 and variance 1 for all features together...");

        float xMean = xTrain.Mean();
        WriteLine($"Current mean: {xMean}. Scale data to mean 0...");
        xTrain.AddInPlace(-xMean);
        xTest.AddInPlace(-xMean);

        WriteLine($"xTrain min: {xTrain.Min()}");
        WriteLine($"xTest min: {xTest.Min()}");
        WriteLine($"xTrain max: {xTrain.Max()}");
        WriteLine($"xTest max: {xTest.Max()}");

        float xStdDev = xTrain.StdDev();
        WriteLine($"\nCurrent stdDev: {xStdDev}. Scale data to variance 1...");
        xTrain.DivideInPlace(xStdDev);
        xTest.DivideInPlace(xStdDev);
        WriteLine($"xTrain min: {xTrain.Min()}");
        WriteLine($"xTest min: {xTest.Min()}");
        WriteLine($"xTrain max: {xTrain.Max()}");
        WriteLine($"xTest max: {xTest.Max()}");

        float yMean = yTrain.Mean();
        float yStdDev = yTrain.StdDev();
        yTrain.AddInPlace(-yMean);
        yTrain.DivideInPlace(yStdDev);
        yTest.AddInPlace(-yMean);
        yTest.DivideInPlace(yStdDev);

        // Create model
        SineFunctionModel model = new(random);

        // Create trainer
        LearningRate learningRate = new ExponentialDecayLearningRate(0.01f, 0.005f);
        Trainer<float[,], float[,]> trainer = new(
            model,
            new AdamOptimizer(learningRate),
            random: random,
            logger: Program.LoggerFactory.CreateLogger<SineFunction>()
        );

        // Train model
        trainer.Fit(
            dataSource,
            epochs: 1_000,
            evalEveryEpochs: 100,
            logEveryEpochs: 100,
            batchSize: 250
        );

        WriteLine();
        WriteLine("Sample predictions vs actual values:");
        WriteLine();
        WriteLine($"{"Sample No",14}{"Predicted",14}{"Actual",14}");
        WriteLine();

        // Show predictions for the test set

        int[] showTestSamples = { 0, 1, 2, testSize - 3, testSize - 2, testSize - 1 };

        // Do a forward pass for all test samples at once

        float[,] predictions = model.Forward(xTest, true);
        foreach (int sampleIndex in showTestSamples)
        {
            //float predictedValue = predictions[sampleIndex, 0];
            //float actualValue = yTest![sampleIndex, 0];

            float predictedValue = predictions[sampleIndex, 0] * yStdDev + yMean;
            float actualValue = yTest[sampleIndex, 0] * yStdDev + yMean;

            WriteLine($"{sampleIndex + 1,14}{predictedValue,14:F4}{actualValue,14:F4}");
        }

    }
}