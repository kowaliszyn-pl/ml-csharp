// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.InteropServices;

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

using Serilog;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;

namespace Autoencoder;

internal class AutoencoderConvModel(int bottleneckDim, SeededRandom? random, string? modelFilePath = null)
    : BaseModel<float[,,,], float[,,,]>(new MeanSquaredErrorLoss4D(MseReduction.ElementMean), random, modelFilePath)
{

    private const int InnerChannels = 3; // 7;
    private const int ImageInnerSize = 28;
    private Layer<float[,], float[,]>? _bottleneckLayer;
    private Layer<float[,], float[,]>? _firstDecoderLayer;

    protected override LayerListBuilder<float[,,,], float[,,,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);

        _bottleneckLayer = new DenseLayer(bottleneckDim, new Tanh2D(), initializer);
        _firstDecoderLayer = new DenseLayer(ImageInnerSize * ImageInnerSize * InnerChannels, new Tanh2D(), initializer);

        return
            // 1. Encoder
            //AddLayer(new Conv2DLayer(
            //    kernels: 14,
            //    kernelHeight: 5,
            //    kernelWidth: 5,
            //    activationFunction: new Tanh4D(),
            //    paramInitializer: initializer
            //))
            AddLayer(new Conv2DLayer(
                kernels: InnerChannels,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ))
            .AddLayer(new FlattenLayer()) // dense1 in encoder

            // 2. Bottleneck
            .AddLayer(_bottleneckLayer)

            // 3. Decoder
            .AddLayer(_firstDecoderLayer)  // dense1 in decoder
            .AddLayer(new UnflattenLayer(InnerChannels, ImageInnerSize, ImageInnerSize))
            //.AddLayer(new Conv2DLayer(
            //    kernels: 14,
            //    kernelHeight: 5,
            //    kernelWidth: 5,
            //    activationFunction: new Tanh4D(),
            //    paramInitializer: initializer
            //))
            .AddLayer(new Conv2DLayer(
                kernels: 1,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ));
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

internal class AutoencoderDenseModel(int bottleneckDim, SeededRandom? random, string? modelFilePath = null)
    : BaseModel<float[,], float[,]>(new MeanSquaredErrorLoss(MseReduction.ElementMean), random, modelFilePath)
{
    private const int InnerChannels = 7;
    private const int ImageInnerSize = 28;
    private Layer<float[,], float[,]>? _bottleneckLayer;
    private Layer<float[,], float[,]>? _firstDecoderLayer;

    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);

        _bottleneckLayer = new DenseLayer(bottleneckDim, new Tanh2D(), initializer);
        _firstDecoderLayer = new DenseLayer(ImageInnerSize * ImageInnerSize * InnerChannels, new Tanh2D(), initializer);

        return AddLayer(new DenseLayer(InnerChannels * ImageInnerSize * ImageInnerSize, new Tanh2D(), initializer))
            .AddLayer(_bottleneckLayer)
            .AddLayer(_firstDecoderLayer)
            .AddLayer(new DenseLayer(ImageInnerSize * ImageInnerSize, new Tanh2D(), initializer));
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

internal class Program
{
    private const int BottleneckDim1 = 24;
    private const int BottleneckDim2 = 28;
    private const int BottleneckDim3 = 56;

    private const int RandomSeed = 260423;
    private const int Epochs = 6;
    private const int BatchSize = 100;
    // private const int EvalEveryEpochs = 2;
    private const int LogEveryEpochs = 1;

    private const float InitialLearningRate = 0.01f;
    private const float FinalLearningRate = 0.0005f;
    private const float AdamBeta1 = 0.89f;
    private const float AdamBeta2 = 0.99f;

    private const string ModelName = "Autoencoder";

    private static Microsoft.Extensions.Logging.ILogger s_logger = default!;

    private static void Main()
    {
        // Create ILogger using Serilog
        Serilog.Core.Logger serilog = new LoggerConfiguration()
            .WriteTo.File("..\\..\\..\\Logs\\log-.txt", rollingInterval: RollingInterval.Day)
            .CreateLogger();

        Log.Logger = serilog;
        Log.Information("Logging started...");

        // Create a LoggerFactory and add Serilog
        ILoggerFactory loggerFactory = new LoggerFactory()
            .AddSerilog(serilog);
        s_logger = loggerFactory.CreateLogger<AutoencoderConvModel>();

        bool running = true;
        OutputEncoding = System.Text.Encoding.UTF8;
        OperationBackend.Use(OperationBackendType.CpuArrays);

        while (running)
        {
            bool fromSubmenu = false;
            WriteLine("Select a routine to run (Autoencoder):");
            WriteLine("B. Select operation backend");
            WriteLine("T. Train and save a model");
            WriteLine("L. Load the last model");
            WriteLine("Other: Exit");
            WriteLine();
            Write("Enter your choice: ");

            string? choice = ReadLine();
            WriteLine();

            switch (choice?.ToUpper())
            {
                case "B":
                    SelectOperationBackend();
                    WriteLine();
                    fromSubmenu = true;
                    break;
                case "T":
                    SelectTrain();
                    fromSubmenu = true;
                    break;
                case "L":
                    SelectLoad();
                    fromSubmenu = true;
                    break;

                default:
                    WriteLine("Goodbye!");
                    running = false;
                    break;
            }

            if (running && !fromSubmenu)
            {
                WriteLine("\nPress any key to continue...");
                ReadKey();
                WriteLine();
            }
        }
    }

    private static void SelectOperationBackend()
    {
        WriteLine("Select operation backend:");
        WriteLine("A. CPU - Arrays");
        WriteLine("S. CPU - Spans");
        WriteLine("P. CPU - Spans Parallel");
        WriteLine("G. GPU");
        WriteLine("Other: Exit");
        WriteLine();
        Write("Enter your choice: ");
        string? backendChoice = ReadLine();
        WriteLine();
        switch (backendChoice?.ToUpper())
        {
            case "A":
                OperationBackend.Use(OperationBackendType.CpuArrays);
                WriteLine("Using CPU - Arrays backend.");
                break;
            case "S":
                OperationBackend.Use(OperationBackendType.CpuSpans);
                WriteLine("Using CPU - Spans backend.");
                break;
            case "P":
                OperationBackend.Use(OperationBackendType.CpuSpansParallel);
                WriteLine("Using CPU - Spans Parallel backend.");
                break;
            case "G":
                OperationBackend.Use(OperationBackendType.Gpu);
                WriteLine("Using GPU backend.");
                break;
            default:
                WriteLine("No changes made to the operation backend.");
                break;
        }
    }

    private static void SelectTrain()
    {
        WriteLine("Select the bottleneck dim to train:");
        WriteLine($"1. {BottleneckDim1}");
        WriteLine($"2. {BottleneckDim2}");
        WriteLine($"3. {BottleneckDim3}");
        WriteLine("Other: Exit");
        WriteLine();
        Write("Enter your choice: ");
        string? trainChoice = ReadLine();
        WriteLine();
        switch (trainChoice?.ToUpper())
        {
            case "1":
                Train(BottleneckDim1);
                break;
            case "2":
                Train(BottleneckDim2);
                break;
            case "3":
                Train(BottleneckDim3);
                break;
            default:
                WriteLine("Back to menu");
                break;
        }
    }

    private static void SelectLoad()
    {
        WriteLine("Select the bottleneck dim to load:");
        WriteLine($"1. {BottleneckDim1}");
        WriteLine($"2. {BottleneckDim2}");
        WriteLine($"3. {BottleneckDim3}");
        WriteLine("Other: Exit");
        WriteLine();
        Write("Enter your choice: ");
        string? loadChoice = ReadLine();
        WriteLine();
        switch (loadChoice?.ToUpper())
        {
            case "1":
                Load(BottleneckDim1);
                break;
            case "2":
                Load(BottleneckDim2);
                break;
            case "3":
                Load(BottleneckDim3);
                break;
            default:
                WriteLine("Back to menu");
                break;
        }
    }

    private static void Train(int bottleneckDim)
    {
        WriteLine("Loading and preprocessing data...");

        (float[,,,] xTrain, _) = LoadAndNormalizeImages();

        // Create another identical pair of xTrain, xTest called yTrain, yTest which will be used as the target output for the autoencoder. The autoencoder will learn to reconstruct the input data, so the target output is the same as the input data.
        // We need them separated because the Trainer shuffles the data and creates batches - in case of having the same array pointers for input and target output, shuffling would break the correspondence between input and target output.

        float[,,,] yTrain = new float[xTrain.GetLength(0), xTrain.GetLength(1), xTrain.GetLength(2), xTrain.GetLength(3)];

        ReadOnlySpan<float> xTrainSpan = MemoryMarshal.CreateReadOnlySpan(ref xTrain[0, 0, 0, 0], xTrain.Length);
        Span<float> yTrainSpan = MemoryMarshal.CreateSpan(ref yTrain[0, 0, 0, 0], yTrain.Length);

        // Copy using Span.CopyTo for better performance
        xTrainSpan.CopyTo(yTrainSpan);

        WriteLine("Creating the model...");

        SimpleDataSource<float[,,,], float[,,,]> dataSource = new(xTrain, yTrain);
        SeededRandom commonRandom = new(RandomSeed);
        AutoencoderConvModel model = new(bottleneckDim, commonRandom);
        LearningRate learningRate = new ExponentialDecayLearningRate(InitialLearningRate, FinalLearningRate, 10);
        // MeanSquaredErrorLoss4D lossFunction = new();

        Trainer<float[,,,], float[,,,]> trainer = new(
            model,
            new AdamOptimizer(learningRate, AdamBeta1, AdamBeta2),
            random: commonRandom,
            logger: s_logger
        )
        {
            Memo = $"Calling class: {nameof(AutoencoderConvModel)}."
        };

        trainer.Fit(
            dataSource,
            //lossFunction: lossFunction,
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

    private static void Load(int bottleneckDim)
    {
        string modelPath = GetFileName(bottleneckDim);
        AutoencoderConvModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);
        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Model parameters loaded from {modelPath}.");
        ResetColor();

        WriteLine("Loading and preprocessing data...");

        (float[,,,] xTrain, float[,] xTrain2D) = LoadAndNormalizeImages();

        float[,,,] yTrain = model.Forward(xTrain, true);

        // Denormalize the output from [-1, 1] back to [0, 255] and convert it to float[row, pixelIndex] for visualization using SaveMnistPicture(int size, int index, float[,] mnistData, string fileName)

        const float scale = 255f / 2f;

        int rows = yTrain.GetLength(0);
        int channels = yTrain.GetLength(1);
        int imageHeight = yTrain.GetLength(2);
        int imageWidth = yTrain.GetLength(3);

        float[,] yTrain2D = new float[rows, channels * imageHeight * imageWidth];

        for (int row = 0; row < rows; row++)
        {
            for (int channel = 0; channel < channels; channel++)
            {
                for (int height = 0; height < imageHeight; height++)
                {
                    for (int width = 0; width < imageWidth; width++)
                    {
                        float normalizedValue = yTrain[row, channel, height, width]; // -1..1
                        float denormalizedValue = (normalizedValue + 1f) * scale; // 0..255
                        yTrain2D[row, channel * imageHeight * imageWidth + height * imageWidth + width] = denormalizedValue;
                    }
                }
            }
        }

        // Now we have xTrain2D and yTrain2D, which can be used for the following visualizations

        WriteLine($"Saving original and reconstructed images from {xTrain2D.Length} xTrain points and {yTrain2D.Length} yTrain points.");

        int[] selectedImages = [20, 21, 22, 23, 30];

        foreach (int index in selectedImages)
        {
            Utils.Drawing.SaveMnistPicture(100, index, xTrain2D, $"model{bottleneckDim}_original_{index}");
            Utils.Drawing.SaveMnistPicture(100, index, yTrain2D, $"model{bottleneckDim}_reconstructed_{index}");
        }

        WriteLine();
    }

    private static (float[,,,] xMerged, float[,] xMerged2D) LoadAndNormalizeImages()
    {
        // Get both (train and test) datasets and merge them into one array.

        float[,] train = LoadCsv("..\\..\\..\\..\\..\\data\\mnist\\mnist_train_small.csv");
        float[,] test = LoadCsv("..\\..\\..\\..\\..\\data\\mnist\\mnist_test.csv");

        int trainRows = train.GetLength(0);
        int testRows = test.GetLength(0);
        int features = train.GetLength(1);

        float[,] merged = new float[trainRows + testRows, features];

        for (int row = 0; row < trainRows; row++)
        {
            for (int col = 0; col < features; col++)
            {
                merged[row, col] = train[row, col];
            }
        }

        for (int row = 0; row < testRows; row++)
        {
            for (int col = 0; col < features; col++)
            {
                merged[trainRows + row, col] = test[row, col];
            }
        }

        // Split the merged data into xMerged and yMerged arrays. The first column of the merged array is used to create a one-hot encoded yMerged array, while the remaining columns are used to create the xMerged array. The xMerged array is then reshaped into a 4D array (xMerged) with dimensions corresponding to the number of samples, channels, height, and width.

        (float[,,,] xMerged, float[,] yMerged, float[,] xMerged2D) = Split(merged);

        // Convert pixel values from [0, 255] to [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1].

        const float min = 0;
        const float max = 255f;
        const float scale = 2f / (max - min); // Scale to range [-1, 1]

        int rows = xMerged.GetLength(0);
        int channels = xMerged.GetLength(1);
        int imageHeight = xMerged.GetLength(2);
        int imageWidth = xMerged.GetLength(3);

        for (int row = 0; row < rows; row++)
        {
            for (int channel = 0; channel < channels; channel++)
            {
                for (int height = 0; height < imageHeight; height++)
                {
                    for (int width = 0; width < imageWidth; width++)
                    {
                        xMerged[row, channel, height, width] = (xMerged[row, channel, height, width] - min) * scale - 1f;
                    }
                }
            }
        }

        return (xMerged, xMerged2D);
    }

    /// <summary>
    /// Splits the input 2D array into xData4D, yData, and xData2D arrays. The first column of the input array is used to create a one-hot encoded yData array, while the remaining columns are used to create the xData2D array. The xData2D array is then reshaped into a 4D array (xData4D) with dimensions corresponding to the number of samples, channels, height, and width.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    private static (float[,,,] xData4D, float[,] yData, float[,] xData2D) Split(float[,] source)
    {
        // Split into xTest (all columns except the first one) and yTest (a one-hot table from the first column with values from 0 to 9).

        float[,] xData2D = source.GetColumns(1..source.GetLength(1));
        float[,] yData = source.GetColumn(0);

        Debug.Assert(xData2D.GetLength(1) == 28 * 28);

        // Convert yTest to a one-hot table.
        int yTestRows = yData.GetLength(0);
        float[,] oneHot = new float[yTestRows, 10];
        for (int row = 0; row < yTestRows; row++)
        {
            int value = Convert.ToInt32(yData[row, 0]);
            oneHot[row, value] = 1f;
        }

        int xDataRows = xData2D.GetLength(0);
        int xDataCols = xData2D.GetLength(1);
        float[,,,] xData4D = new float[xDataRows, 1, 28, 28];

        for (int row = 0; row < xDataRows; row++)
        {
            for (int col = 0; col < xDataCols; col++)
            {
                //int x = col % 28;
                //int y = col / 28;
                xData4D[row, 0 /* one input channel */, col / 28, col % 28] = xData2D[row, col];
            }
        }

        return (xData4D, oneHot, xData2D);
    }

    private static string GetFileName(int bottleneckDim)
        => $"{ModelName}_{bottleneckDim}.json";
}
