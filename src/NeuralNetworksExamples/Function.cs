// Neural Networks in C♯
// File name: Function.cs
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

namespace NeuralNetworksExamples;

class FunctionModel(SeededRandom? random)
    : Model<float[,], float[,]>(new MeanSquaredError(), random)
{
    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        GlorotInitializer initializer = new(Random);

        return AddLayer(new DenseLayer(4, new ReLU(), initializer))
            //.AddLayer(new DenseLayer(4, new ReLU(), initializer))
            //.AddLayer(new DenseLayer(4, new Sigmoid(), initializer))
            .AddLayer(new DenseLayer(4, new ReLU(), initializer))
            .AddLayer(new DenseLayer(4, new ReLU(), initializer))
            .AddLayer(new DenseLayer(1, new Linear(), initializer));
    }

}

class Function
{
    public static void Run()
    {
        // Create data set
        int sampleCount = 1_000;
        float[,] arguments = new float[sampleCount, 2];
        SeededRandom random = new(251202);
        for (int i = 0; i < sampleCount; i++)
        {
            arguments[i, 0] = random.NextSingle() * 20 - 10;
            arguments[i, 1] = random.NextSingle() * 20 - 10;
        }
        //Func<float[], float> function = (float[] args) => MathF.Sin(args[0]) + MathF.Cos(args[1]);
        Func<float[], float> function = (float[] args) => 5 * args[0] - 3 * args[1] * args[1]  + 1.4f;
        //Func<float[], float> function = (float[] x) => 2f * x[0] * x[0] - 6f * x[1] + 4.5f;
        var dataSource = new FunctionDataSource(arguments, function, 0.7f, random);
        (float[,] xTrain, float[,] yTrain, float[,]? xTest, float[,]? yTest) = dataSource.GetData();
        int nTest = xTest!.GetLength(0);

        // Create model
        FunctionModel model = new(random);

        // Create trainer
        LearningRate learningRate = new ExponentialDecayLearningRate(0.01f, 0.005f);
        var trainer = new Trainer2D(
            model,
            new StochasticGradientDescentMomentum(learningRate, 0.9f),
            random,
            Program.LoggerFactory.CreateLogger<Trainer2D>()
        );

        // Train model
        trainer.Fit(
            dataSource,
            epochs: 16_000,
            evalEveryEpochs: 1_000,
            logEveryEpochs: 1_000,
            batchSize: 250
        );

        WriteLine();
        WriteLine("Sample predictions vs actual values:");
        WriteLine();
        WriteLine($"{"Sample No",14}{"Predicted",14}{"Actual",14}");
        WriteLine();

        // Show predictions for the test set

        int[] showTestSamples = { 0, 1, 2, nTest - 3, nTest - 2, nTest - 1 };

        // Do a forward pass for all test samples at once

        float[,] predictions = model.Forward(xTest, true);
        foreach (int sampleIndex in showTestSamples)
        {
            float predictedValue = predictions[sampleIndex, 0];
            float actualValue = yTest![sampleIndex, 0];
            WriteLine($"{sampleIndex + 1,14}{predictedValue,14:F4}{actualValue,14:F4}");
        }

    }
}