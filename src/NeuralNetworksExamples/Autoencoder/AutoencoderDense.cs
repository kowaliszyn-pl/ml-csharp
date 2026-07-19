// Neural Networks in C♯
// File name: AutoencoderDense.cs
// www.kowaliszyn.pl, 2025 - 2026

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

using static System.Console;
using static NeuralNetworksExamples.Autoencoder.Utils;
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
            AddLayer(new DenseLayer(178, new LeakyReLU2D(), initializer))
            .AddLayer(new DenseLayer(46, new LeakyReLU2D(), initializer))

            // Bottleneck
            .AddLayer(_bottleneckLayer = new DenseLayer(bottleneckDim, new Softsign(), initializer))

            // Decoder
            .AddLayer(_firstDecoderLayer = new DenseLayer(46, new LeakyReLU2D(), initializer))
            .AddLayer(new DenseLayer(178, new LeakyReLU2D(), initializer))
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
    private const int Epochs = 30;
    private const int BatchSize = 400;
    private const int LogEveryEpochs = 1;

    private const float InitialLearningRate = 1e-2f;
    private const float FinalLearningRate = 7e-3f;
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

        TanhNormalizeInPlace(xTrain);
        TanhNormalizeInPlace(xTest);

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

        string modelPath = GetFileName(ModelName, bottleneckDim);
        model.SaveParams(modelPath, "Final trained model.");
        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Model parameters saved to {modelPath}.");
        ResetColor();
    }

    internal static void Load()
    {
        int bottleneckDim = Program.LatentSpaceDimensions;
        string modelPath = GetFileName(ModelName, bottleneckDim);
        AutoencoderDenseModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);
        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Model parameters loaded from {modelPath}.");
        ResetColor();

        WriteLine("Loading and preprocessing data...");

        float[,] xTrain = LoadTrainingData();

        float[,] originalImages = (float[,])xTrain.Clone();

        // Normalize the pixel values from [0, 255] to [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1].

        TanhNormalizeInPlace(xTrain);

        WriteLine("Reconstructing images using the loaded model...");

        float[,] reconstructedImages = model.Forward(xTrain, true);

        // Rescale the pixel values back to [0, 255] for visualization purposes.

        DenormalizeToPixelValuesInPlace(reconstructedImages);

        // Generate random encoded data for visualization of the decoder's output
        float[,] randomEncoded = GenerateRandomEncodedData(bottleneckDim, 5, RandomSeed);
        float[,] randomlyGeneratedImages = model.Decode(randomEncoded);
        DenormalizeToPixelValuesInPlace(randomlyGeneratedImages);

        // Now we have xTrain2D and yTrain2D, which can be used for the following visualizations

        SaveReconstructionComparison(ModelName, bottleneckDim, originalImages, reconstructedImages, randomlyGeneratedImages);
    }

    internal static void VisualizeLatentSpace()
    {
        int bottleneckDim = Program.LatentSpaceDimensions;

        WriteLine("Loading model and data...");

        string modelPath = GetFileName(ModelName, bottleneckDim);
        AutoencoderDenseModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);

        // Load data and labels
        float[,] train = GetMnistTrainData();

        // Restrict to MaxSamplesToVisualize samples for t-SNE visualization to reduce computation time
        train = train.GetRows(0..Program.MaxSamplesToVisualize);

        (float[,] xTrain, float[,] labels) = SplitFeaturesAndLabels(train);

        // Normalize
        TanhNormalizeInPlace(xTrain);

        // Get latent representation
        WriteLine("Encoding data to latent space...");
        _ = model.Forward(xTrain, false);
        float[,] encoded = model.GetEncodedRepresentation();
        VisualizeWithHistogramAndTSNE(ModelName, labels, encoded);
    }

    private static float[,] LoadTrainingData()
        => ExtractFeatureColumns(GetMnistTrainData());

    private static float[,] LoadTestData()
        => ExtractFeatureColumns(GetMnistTestData());
}
