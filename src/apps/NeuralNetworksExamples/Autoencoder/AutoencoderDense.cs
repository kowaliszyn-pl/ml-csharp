// Neural Networks in C♯
// File name: AutoencoderDense.cs
// www.kowaliszyn.pl, 2025 - 2026

using Accord.MachineLearning.Clustering;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.Core.Operations;
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

using ScottPlot;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;
using static NeuralNetworksExamples.Utils;

namespace NeuralNetworksExamples.Autoencoder;

internal class AutoencoderDenseModel(int bottleneckDim, SeededRandom? random, string? modelFilePath = null)
    : BaseModel<float[,], float[,]>(new MeanSquaredErrorLoss(MseReduction.ElementMean), random, modelFilePath)
{
    private Layer<float[,], float[,]>? _bottleneckLayer;
    private Layer<float[,], float[,]>? _firstDecoderLayer;

    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);

        return
            // Encoder
            AddLayer(new DenseLayer(178, new LeakyReLU2D(), initializer, new Dropout2D(0.8f, Random)))
            .AddLayer(new DenseLayer(46, new LeakyReLU2D(), initializer, new Dropout2D(0.8f, Random)))

            // Bottleneck
            .AddLayer(_bottleneckLayer = new DenseLayer(bottleneckDim, new Linear(), initializer))

            // Decoder
            .AddLayer(_firstDecoderLayer = new DenseLayer(46, new LeakyReLU2D(), initializer, new Dropout2D(0.8f, Random)))
            .AddLayer(new DenseLayer(178, new LeakyReLU2D(), initializer, new Dropout2D(0.8f, Random)))
            .AddLayer(new DenseLayer(784, new Tanh2D(), initializer));
    }

    /// <summary>
    /// Gets the encoded representation (latent data) produced by the bottleneck layer of the model.
    /// </summary>
    /// <returns>
    /// A two-dimensional array of floating-point values representing the output of the bottleneck layer.
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown if the bottleneck layer output is not available.</exception>
    public float[,] GetEncodedRepresentation()
    {
        return _bottleneckLayer?.Output
            ?? throw new InvalidOperationException("Bottleneck layer output is not available.");
    }

    /// <summary>
    /// Forward encoded representation and return the decoded output. This can be used to visualize the output of the
    /// decoder part of the autoencoder based on randomly generated encoded data or to see how the decoder reconstructs
    /// the input data from the encoded (bottleneck) representation.
    /// </summary>
    public float[,] Decode(float[,] encoded)
    {
        // We need to pass the encoded data through the first decoder layer and then through the remaining layers of the model.

        if (_firstDecoderLayer is null)
            throw new InvalidOperationException("Decoder layer is not initialized.");

        return InferFromLayer(_firstDecoderLayer, encoded);
    }
}

internal class AutoencoderDense
{
    private const int RandomSeed = 260710;
    private const int Epochs = 10;
    private const int BatchSize = 400;
    private const int LogEveryEpochs = 1;

    private const float InitialLearningRate = 0.01f;
    private const float FinalLearningRate = 0.0005f;
    private const float AdamBeta1 = 0.89f;
    private const float AdamBeta2 = 0.99f;

    private const string ModelName = "AutoencoderDense";

    internal static void Train()
    {
        ILogger logger = Program.LoggerFactory.CreateLogger<AutoencoderDense>();
        int bottleneckDim = Program.LatentSpaceDimensions;

        WriteLine("Loading and preprocessing data...");

        float[,] xTrain = LoadTrainingData();
        float[,] xTest = LoadTestData();

        // Save a copy of the training images for drawing purposes before normalization.

        float[,] trainingImagesForDrawing = (float[,])xTrain.Clone();

        // Normalize the pixel values from [0, 255] to [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1].

        NormalizeToTanhRange(xTrain);
        NormalizeToTanhRange(xTest);

        float[,] yTrain = (float[,])xTrain.Clone();

        // It's not quite necessary to clone the test data, but we do it for consistency.
        float[,] yTest = (float[,])xTest.Clone();

        WriteLine("Creating the model...");

        SimpleDataSource<float[,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);
        SeededRandom commonRandom = new(RandomSeed);
        AutoencoderDenseModel model = new(bottleneckDim, commonRandom);
        LearningRate learningRate = new ExponentialDecayLearningRate(InitialLearningRate, FinalLearningRate, 10);

        Trainer<float[,], float[,]> trainer = new(
            model,
            new AdamOptimizer(learningRate, AdamBeta1, AdamBeta2),
            random: commonRandom,
            logger: logger
        )
        {
            Memo = $"Calling class: {nameof(AutoencoderDenseModel)}."
        };

        trainer.Fit(
            dataSource,
            epochs: Epochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize,
            saveParamsOnBestLoss: false,
            showLossOnStart: false
        );

        WriteLine("Training completed.");

        // Save the model

        string modelPath = GetFileName(bottleneckDim);
        model.SaveParams(modelPath, "Final trained model.");
        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Model parameters saved to {modelPath}.");
        ResetColor();
        WriteLine();
    }

    internal static void Load()
    {
        int bottleneckDim = Program.LatentSpaceDimensions;
        string modelPath = GetFileName(bottleneckDim);
        AutoencoderDenseModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);
        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Model parameters loaded from {modelPath}.");
        ResetColor();

        WriteLine("Loading and preprocessing data...");

        float[,] xTrain = LoadTrainingData();

        WriteLine($"Loaded {xTrain.GetLength(0)} training samples with {xTrain.GetLength(1)} features each.");

        float[,] trainingImagesForDrawing = (float[,])xTrain.Clone();

        // Normalize the pixel values from [0, 255] to [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1].

        NormalizeToTanhRange(xTrain);

        float[,] yTrain = model.Forward(xTrain, true);

        // Rescale the pixel values back to [0, 255] for visualization purposes.

        RescaleToPixelValues(yTrain);

        // Now we have xTrain2D and yTrain2D, which can be used for the following visualizations

        WriteLine($"Saving original and reconstructed images from {xTrain.Length} xTrain points and {yTrain.Length} yTrain points.");

        int[] selectedImages = [20, 21, 22, 23, 30];

        foreach (int index in selectedImages)
        {
            Drawing.SaveMnistPicture(100, index, trainingImagesForDrawing, $"{ModelName}_{bottleneckDim}_original_{index}");
            Drawing.SaveMnistPicture(100, index, yTrain, $"{ModelName}_{bottleneckDim}_reconstructed_{index}");
        }

        WriteLine();
    }

    internal static void VisualizeLatentSpace()
    {
        int bottleneckDim = Program.LatentSpaceDimensions;

        WriteLine("Loading model and data...");

        string modelPath = GetFileName(bottleneckDim);
        AutoencoderDenseModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);

        // Load data and labels
        float[,] train = LoadCsv("..\\..\\..\\..\\..\\data\\MNIST\\mnist_train_small.csv");

        // Restrict to 10000 samples for t-SNE visualization to reduce computation time
        int maxSamples = 15000;
        if (train.GetLength(0) > maxSamples)
        {
            train = train.GetRows(0..maxSamples);
        }

        float[,] labels = train.GetColumn(0);
        float[,] xTrain = train.GetColumns(1..train.GetLength(1));

        // Normalize
        NormalizeToTanhRange(xTrain);

        // Get latent representation
        WriteLine("Encoding data to latent space...");
        _ = model.Forward(xTrain, false);
        float[,] encoded = model.GetEncodedRepresentation();

        // Convert to double[][] for Accord.NET
        int n = encoded.GetLength(0);
        int dim = encoded.GetLength(1);
        double[][] encodedDouble = new double[n][];
        for (int i = 0; i < n; i++)
        {
            encodedDouble[i] = new double[dim];
            for (int j = 0; j < dim; j++)
            {
                encodedDouble[i][j] = encoded[i, j];
            }
        }

        // Apply t-SNE
        WriteLine("Applying t-SNE reduction...");
        TSNE tsne = new()
        {
            NumberOfOutputs = 2,
            Perplexity = 30,
            //Iterations = 1000
        };

        double[][] reduced = tsne.Transform(encodedDouble);

        // Create plot
        WriteLine("Creating visualization...");
        Plot plt = new();

        // Group points by digit
        for (int digit = 0; digit <= 9; digit++)
        {
            List<double> xPoints = [];
            List<double> yPoints = [];

            for (int i = 0; i < n; i++)
            {
                if ((int)labels[i, 0] == digit)
                {
                    xPoints.Add(reduced[i][0]);
                    yPoints.Add(reduced[i][1]);
                }
            }

            var scatter = plt.Add.ScatterPoints(xPoints, yPoints);
            scatter.Label = $"Digit {digit}";
            scatter.MarkerSize = 5;
        }

        plt.ShowLegend();
        plt.Title($"t-SNE Visualization of Latent Space (bottleneck={bottleneckDim})");
        plt.XLabel("t-SNE Component 1");
        plt.YLabel("t-SNE Component 2");

        string outputPath = $"..\\..\\..\\{ModelName}_{bottleneckDim}_tsne.png";
        plt.SavePng(outputPath, 1200, 900);

        ForegroundColor = ConsoleColor.Green;
        WriteLine($"t-SNE plot saved to {outputPath}");
        ResetColor();
        WriteLine();
    }

    private static float[,] LoadTrainingData()
    {
        float[,] train = GetMnistTrainData();
        (float[,] xTrain, _) = Split(train);
        return xTrain;
    }

    private static float[,] LoadTestData()
    {
        float[,] test = GetMnistTestData();
        (float[,] xTest, _) = Split(test);
        return xTest;
    }

    private static void NormalizeToTanhRange(float[,] xTrain)
    {
        const float min = 0;
        const float max = 255f;
        const float scale = 2f / (max - min); // Scale to range [-1, 1]

        for (int row = 0; row < xTrain.GetLength(0); row++)
        {
            for (int col = 0; col < xTrain.GetLength(1); col++)
            {
                xTrain[row, col] = (xTrain[row, col] - min) * scale - 1f;
            }
        }
    }

    private static void RescaleToPixelValues(float[,] yTrain)
    {
        const float scale = 255f / 2f;

        for (int row = 0; row < yTrain.GetLength(0); row++)
        {
            for (int col = 0; col < yTrain.GetLength(1); col++)
            {
                yTrain[row, col] = (yTrain[row, col] + 1f) * scale;
            }
        }
    }

    private static (float[,] xData, float[,] yData) Split(float[,] source)
    {
        // Split into xData (all columns except the first one) and yData (a one-hot table from the first column with values from 0 to 9).

        float[,] xData = source.GetColumns(1..source.GetLength(1));
        float[,] yData = source.GetColumn(0);

        // Convert yData to a one-hot table.
        float[,] oneHot = new float[yData.GetLength(0), 10];
        for (int row = 0; row < yData.GetLength(0); row++)
        {
            int value = Convert.ToInt32(yData[row, 0]);
            oneHot[row, value] = 1f;
        }

        return (xData, oneHot);
    }

    private static string GetFileName(int bottleneckDim)
        => $"{ModelName}_{bottleneckDim}.json";
}
