// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

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

internal class AutoencoderModel(int bottleneckDim, SeededRandom? random, string? modelFilePath = null) : BaseModel<float[,,,], float[,,,]>(null, random, modelFilePath)
{

    private const int InnerChannels = 7;
    private const int ImageInnerSize = 28;
    private Layer<float[,], float[,]>? _bottleneckLayer;
    private Layer<float[,], float[,]>? _firstDecoderLayer;

    protected override LayerListBuilder<float[,,,], float[,,,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);

        _bottleneckLayer = new DenseLayer(bottleneckDim, new Linear(), initializer);
        _firstDecoderLayer = new DenseLayer(ImageInnerSize * ImageInnerSize * InnerChannels, new Linear(), initializer);

        return
            AddLayer(new Conv2DLayer(
                kernels: 14,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ))
            .AddLayer(new Conv2DLayer(
                kernels: 7,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ))
            .AddLayer(new FlattenLayer())
            .AddLayer(_bottleneckLayer)
            .AddLayer(_firstDecoderLayer)
            .AddLayer(new UnflattenLayer(InnerChannels, ImageInnerSize, ImageInnerSize))
            .AddLayer(new Conv2DLayer(
                kernels: 14,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ))
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

internal class Program
{
    private const int RandomSeed = 260423;
    private const int BottleneckDim = 28;
    private const int Epochs = 5;
    private const int BatchSize = 400;
    // private const int EvalEveryEpochs = 2;
    private const int LogEveryEpochs = 1;

    private const float InitialLearningRate = 0.002f;
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
        s_logger = loggerFactory.CreateLogger<AutoencoderModel>();

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
                    Train();
                    break;
                case "L":
                    // Load();
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

    private static void Train()
    {
        WriteLine("Loading and preprocessing data...");

        // rows - batch
        // cols - features
        float[,] train = LoadCsv("..\\..\\..\\..\\..\\data\\mnist\\mnist_train_small.csv");
        float[,] test = LoadCsv("..\\..\\..\\..\\..\\data\\mnist\\mnist_test.csv");

        (float[,,,] xTrain, _, _) = Split(train);
        (float[,,,] xTest, _, float[,] testImagesForDrawing) = Split(test);

        int trainRows = xTrain.GetLength(0);
        int testRows = xTest.GetLength(0);

        int channels = xTrain.GetLength(1);
        int imageHeight = xTrain.GetLength(2);
        int imageWidth = xTrain.GetLength(3);

        // Convert pixel values from [0, 255] to [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1].

        const float min = 0;
        const float max = 255f;
        const float scale = 2f / (max - min); // Scale to range [-1, 1]

        for (int channel = 0; channel < channels; channel++)
        {
            for (int height = 0; height < imageHeight; height++)
            {
                for (int width = 0; width < imageWidth; width++)
                {

                    for (int row = 0; row < trainRows; row++)
                    {
                        xTrain[row, channel, height, width] = (xTrain[row, channel, height, width] - min) * scale - 1f;
                    }
                    for (int row = 0; row < testRows; row++)
                    {
                        xTest[row, channel, height, width] = (xTest[row, channel, height, width] - min) * scale - 1f;
                    }
                }
            }
        }

        // Now create another identical pair of xTrain, xTest called yTrain, yTest which will be used as the target output for the autoencoder. The autoencoder will learn to reconstruct the input data, so the target output is the same as the input data.
        // We need them separated becaude the Trainer shuffles the data and creates batches - in case of having the same array pointers for input and target output, shuffling would break the correspondence between input and target output.

        float[,,,] yTrain = new float[trainRows, channels, imageHeight, imageWidth];
        float[,,,] yTest = new float[testRows, channels, imageHeight, imageWidth];

        // Copy the values

        for (int channel = 0; channel < channels; channel++)
        {
            for (int height = 0; height < imageHeight; height++)
            {
                for (int width = 0; width < imageWidth; width++)
                {
                    for (int row = 0; row < trainRows; row++)
                    {
                        yTrain[row, channel, height, width] = xTrain[row, channel, height, width];
                    }
                    for (int row = 0; row < testRows; row++)
                    {
                        yTest[row, channel, height, width] = xTest[row, channel, height, width];
                    }
                }
            }
        }

        WriteLine("Creating the model...");

        SimpleDataSource<float[,,,], float[,,,]> dataSource = new(xTrain, yTrain, xTest, yTest);
        SeededRandom commonRandom = new(RandomSeed);
        AutoencoderModel model = new(BottleneckDim, commonRandom);
        LearningRate learningRate = new ExponentialDecayLearningRate(InitialLearningRate, FinalLearningRate, 10);
        MeanSquaredErrorLoss4D lossFunction = new();

        Trainer<float[,,,], float[,,,]> trainer = new(
            model,
            new AdamOptimizer(learningRate, AdamBeta1, AdamBeta2),
            random: commonRandom,
            logger: s_logger
        )
        {
            Memo = $"Calling class: {nameof(AutoencoderModel)}."
        };

        trainer.Fit(
            dataSource,
            lossFunction: lossFunction,
            epochs: Epochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize,
            saveParamsOnBestLoss: false,
            showLossOnStart: true
        );

        // Save the model

        string modelPath = $"{ModelName}.json";
        model.SaveParams(modelPath, "Final trained model.");
        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Model parameters saved to {modelPath}.");
        ResetColor();
    }

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
}
