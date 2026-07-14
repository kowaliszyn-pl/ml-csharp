// Neural Networks in C♯
// File name: AutoencoderCnn.cs
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
using NeuralNetworks.Optimizers;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;

using ScottPlot;

using ScottPlot.Plottables;

using static System.Console;
using static NeuralNetworksExamples.Utils;

namespace NeuralNetworksExamples.Autoencoder;

internal class AutoencoderConvModel(int bottleneckDim, SeededRandom? random, string? modelFilePath = null)
    : BaseModel<float[,,,], float[,,,]>(new MeanSquaredErrorLoss4D(MseReduction.ElementMean), random, modelFilePath)
{

    private Layer<float[,], float[,]>? _bottleneckLayer;
    private Layer<float[,], float[,]>? _firstDecoderLayer;

    protected override LayerListBuilder<float[,,,], float[,,,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);

        return
            // 1. Encoder
            // 1 * 28 * 28
            AddLayer(new Conv2DLayer(
                kernels: 32,
                kernelHeight: 3,
                kernelWidth: 3,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ))
            // 32 * 28 * 28
            .AddLayer(new MaxPooling2DLayer(2, 2))
            // 32 * 14 * 14
            .AddLayer(new FlattenLayer())

            // 2. Bottleneck
            // 32 * 14 * 14 = 6272
            .AddLayer(_bottleneckLayer = new DenseLayer(bottleneckDim, new Tanh2D(), initializer))

            // 3. Decoder
            // bottleneckDim
            .AddLayer(_firstDecoderLayer = new DenseLayer(32 * 14 * 14, new LeakyReLU2D(), initializer))
            // 32 * 14 * 14 = 6272 as a flattened representation
            .AddLayer(new UnflattenLayer(32, 14, 14))
            // 32 * 14 * 14
            .AddLayer(new Upsample2DLayer(2, 2))
            // 32 * 28 * 28
            .AddLayer(new Conv2DLayer(
                kernels: 1,
                kernelHeight: 3,
                kernelWidth: 3,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ));

        // 1 * 28 * 28 as output
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
    public float[,,,] Decode(float[,] encoded)
    {
        // We need to pass the encoded data through the first decoder layer and then through the remaining layers of the model.

        if (_firstDecoderLayer is null)
            throw new InvalidOperationException("Decoder layer is not initialized.");

        return InferFromLayer(_firstDecoderLayer, encoded);
    }
}

internal class AutoencoderCnn
{
    private const int RandomSeed = 260423;
    private const int Epochs = 10;
    private const int BatchSize = 200;
    private const int LogEveryEpochs = 1;

    private const float InitialLearningRate = 1e-2f; //0.01f;
    private const float FinalLearningRate = 5e-4f; // 0.0005f;
    private const float AdamBeta1 = 0.89f;
    private const float AdamBeta2 = 0.99f;

    private const string ModelName = "AutoencoderConv";

    internal static void Train()
    {
        ILogger logger = Program.LoggerFactory.CreateLogger<AutoencoderCnn>();
        int bottleneckDim = Program.LatentSpaceDimensions;

        WriteLine("Loading and preprocessing data...");

        float[,] train = GetMnistTrainData();
        float[,,,] xTrain = ExtractFeaturesAsTanhNormalized4D(train);

        float[,,,] yTrain = (float[,,,])xTrain.Clone();

        float[,] test = GetMnistTestData();
        float[,,,] xTest = ExtractFeaturesAsTanhNormalized4D(test);

        // It's not quite necessary to clone the test data, but we do it for consistency.
        float[,,,] yTest = (float[,,,])xTest.Clone();

        WriteLine("Creating the model...");

        SimpleDataSource<float[,,,], float[,,,]> dataSource = new(xTrain, yTrain, xTest, yTest);
        SeededRandom commonRandom = new(RandomSeed);
        AutoencoderConvModel model = new(bottleneckDim, commonRandom);
        LearningRate learningRate = new ExponentialDecayLearningRate(InitialLearningRate, FinalLearningRate, 10);

        Trainer<float[,,,], float[,,,]> trainer = new(
            model,
            new AdamOptimizer(learningRate, AdamBeta1, AdamBeta2),
            random: commonRandom,
            logger: logger
        )
        {
            Memo = $"Calling class: {nameof(AutoencoderConvModel)}."
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
    }

    internal static void Load()
    {
        int bottleneckDim = Program.LatentSpaceDimensions;
        string modelPath = GetFileName(bottleneckDim);
        AutoencoderConvModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);
        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Model parameters loaded from {modelPath}.");
        ResetColor();

        WriteLine("Loading and preprocessing data...");

        float[,] train = GetMnistTrainData();
        float[,] originalImages = ExtractFeatureColumns(train);

        float[,,,] xTrain = TanhNormalizeAndReshapeTo4D(originalImages);

        WriteLine("Reconstructing images using the loaded model...");

        float[,,,] yTrain = model.Forward(xTrain, true);

        float[,] reconstructedImages = DenormalizeAndReshapeTo2D(yTrain);

        // Now we have xTrain2D and yTrain2D, which can be used for the following visualizations

        WriteLine($"Saving original and reconstructed images.");

        int[] selectedImages = [31, 32, 33, 34, 35];

        foreach (int index in selectedImages)
        {
            Drawing.SaveMnistPicture(100, index, originalImages, $"{ModelName}_{bottleneckDim}_original_{index}");
            Drawing.SaveMnistPicture(100, index, reconstructedImages, $"{ModelName}_{bottleneckDim}_reconstructed_{index}");
        }
    }

    internal static void VisualizeLatentSpace()
    {
        int bottleneckDim = Program.LatentSpaceDimensions;

        WriteLine("Loading model and data...");

        string modelPath = GetFileName(bottleneckDim);
        AutoencoderConvModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);

        // Load data and labels
        float[,] train = GetMnistTrainData();

        // Restrict to MaxSamplesToVisualize samples for t-SNE visualization to reduce computation time
        train = train.GetRows(0..Program.MaxSamplesToVisualize);

        float[,] labels = train.GetColumn(0);
        train = ExtractFeatureColumns(train);

        float[,,,] xTrain = TanhNormalizeAndReshapeTo4D(train);

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

            Scatter scatter = plt.Add.ScatterPoints(xPoints, yPoints);
            scatter.LegendText = $"Digit {digit}";
            scatter.MarkerSize = 5;
        }

        plt.ShowLegend();
        plt.Title($"t-SNE Visualization of Latent Space (bottleneck={dim}, points={n})");
        plt.XLabel("t-SNE Component 1");
        plt.YLabel("t-SNE Component 2");

        string outputPath = $"{ModelName}_{dim}_{n}_tsne.png";
        plt.SavePng(outputPath, 1200, 900);

        ForegroundColor = ConsoleColor.Green;
        WriteLine($"t-SNE plot saved to {outputPath}");
        ResetColor();
    }

    private static string GetFileName(int bottleneckDim)
        => $"{ModelName}_{bottleneckDim}.json";
}
