// Neural Networks in C♯
// File name: AutoencoderCnn.cs
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

internal class AutoencoderConvModel(
    int bottleneckDim,
    SeededRandom? random,
    string? modelFilePath = null)
    : AutoencoderModel<float[,,,]>(
        new MeanSquaredErrorLoss4D(MseReduction.ElementMean),
        random,
        modelFilePath)
{

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
            .AddLayer(BottleneckLayer = new DenseLayer(bottleneckDim, new Tanh2D(), initializer))

            // 3. Decoder
            // bottleneckDim
            .AddLayer(FirstDecoderLayer = new DenseLayer(32 * 14 * 14, new LeakyReLU2D(), initializer))
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
            Memo = $"Calling class: {nameof(AutoencoderCnn)}."
        };

        float? finalLoss = trainer.Fit(
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
        model.SaveParams(modelPath, $"Final trained model with loss {finalLoss:F5}. Date time: {DateTime.Now}.");
        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Model parameters saved to {modelPath}.");
        ResetColor();
    }

    internal static void Load()
    {
        int bottleneckDim = Program.LatentSpaceDimensions;
        string modelPath = GetFileName(ModelName, bottleneckDim);
        AutoencoderConvModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);
        string bottleneckActivationFunction = model.GetBottleneckActivationFunction().GetType().Name;

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

        // Generate random encoded data for visualization of the decoder's output
        float[,] randomEncoded = GenerateRandomEncodedData(bottleneckDim, 5, RandomSeed);
        float[,,,] randomDecoded = model.Decode(randomEncoded);
        float[,] randomlyGeneratedImages = DenormalizeAndReshapeTo2D(randomDecoded);

        // Now we have xTrain2D and yTrain2D, which can be used for the following visualizations

        SaveReconstructionComparison(ModelName, bottleneckActivationFunction, bottleneckDim, originalImages, reconstructedImages, randomlyGeneratedImages);
    }

    internal static void VisualizeLatentSpace()
    {
        int bottleneckDim = Program.LatentSpaceDimensions;

        WriteLine("Loading model and data...");

        string modelPath = GetFileName(ModelName, bottleneckDim);
        AutoencoderConvModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);
        string bottleneckActivationFunction = model.GetBottleneckActivationFunction().GetType().Name;

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

        VisualizeWithHistogramAndTSNE(ModelName, bottleneckActivationFunction, labels, encoded);
    }

}
