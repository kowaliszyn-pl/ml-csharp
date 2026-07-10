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
using NeuralNetworks.Operations.Dropouts;
using NeuralNetworks.Optimizers;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;

using Serilog;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;

namespace Autoencoder;

internal class AutoencoderDenseModel(int bottleneckDim, SeededRandom? random, string? modelFilePath = null)
    : BaseModel<float[,], float[,]>(new MeanSquaredErrorLoss(MseReduction.ElementMean), random, modelFilePath)
{
    private Layer<float[,], float[,]>? _bottleneckLayer;
    private Layer<float[,], float[,]>? _firstDecoderLayer;

    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);

        _bottleneckLayer = new DenseLayer(bottleneckDim, new Linear(), initializer);
        _firstDecoderLayer = new DenseLayer(46, new LeakyReLU2D(), initializer, new Dropout2D(0.8f, Random));

        return 
            // Encoder
            AddLayer(new DenseLayer(178, new LeakyReLU2D(), initializer, new Dropout2D(0.8f, Random)))
            .AddLayer(new DenseLayer(46, new LeakyReLU2D(), initializer, new Dropout2D(0.8f, Random)))
            
            // Bottleneck
            .AddLayer(_bottleneckLayer)
            
            // Decoder
            .AddLayer(_firstDecoderLayer)
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

internal class Program
{
    private const int BottleneckDim1 = 24;
    private const int BottleneckDim2 = 28;
    private const int BottleneckDim3 = 32;

    private const int RandomSeed = 260710;
    private const int Epochs = 6;
    private const int BatchSize = 100;
    // private const int EvalEveryEpochs = 2;
    private const int LogEveryEpochs = 1;

    private const float InitialLearningRate = 0.01f;
    private const float FinalLearningRate = 0.0005f;
    private const float AdamBeta1 = 0.89f;
    private const float AdamBeta2 = 0.99f;

    private const string ModelName = "AutoencoderDense";

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
        s_logger = loggerFactory.CreateLogger<Program>();

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

        float[,] train = LoadCsv("..\\..\\..\\..\\..\\data\\MNIST\\mnist_train_small.csv");
        float[,] test = LoadCsv("..\\..\\..\\..\\..\\data\\MNIST\\mnist_test.csv");

        (float[,] xTrain, _) = Split(train);
        (float[,] xTest, _) = Split(test);
        float[,] trainingImagesForDrawing = (float[,])xTrain.Clone();

        // Normalize the pixel values from [0, 255] to [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1].

        const float min = 0;
        const float max = 255f;
        const float scale = 2f / (max - min); // Scale to range [-1, 1]

        for(int row = 0; row < xTrain.GetLength(0); row++)
        {
            for (int col = 0; col < xTrain.GetLength(1); col++)
            {
                xTrain[row, col] = (xTrain[row, col] - min) * scale - 1f;
            }
        }

        float[,] yTrain = (float[,])xTrain.Clone();

        for(int row = 0; row < xTest.GetLength(0); row++)
        {
            for (int col = 0; col < xTest.GetLength(1); col++)
            {
                xTest[row, col] = (xTest[row, col] - min) * scale - 1f;
            }
        }

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
            logger: s_logger
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

    private static void Load(int bottleneckDim)
    {
        string modelPath = GetFileName(bottleneckDim);
        AutoencoderDenseModel model = new(bottleneckDim, new SeededRandom(RandomSeed), modelPath);
        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Model parameters loaded from {modelPath}.");
        ResetColor();

        WriteLine("Loading and preprocessing data...");

        float[,] train = LoadCsv("..\\..\\..\\..\\..\\data\\MNIST\\mnist_train_small.csv");

        (float[,] xTrain, _) = Split(train);
        WriteLine($"Loaded {xTrain.GetLength(0)} training samples with {xTrain.GetLength(1)} features each.");

        float[,] trainingImagesForDrawing = (float[,])xTrain.Clone();

        // Normalize the pixel values from [0, 255] to [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1].

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

        float[,] yTrain = model.Forward(xTrain, true);

        // Rescale the pixel values back to [0, 255] for visualization purposes.

        const float scaleUp = 255f / 2f;

        for(int row = 0; row < yTrain.GetLength(0); row++)
        {
            for (int col = 0; col < yTrain.GetLength(1); col++)
            {
                yTrain[row, col] = (yTrain[row, col] + 1f) * scaleUp;
            }
        }

        // Now we have xTrain2D and yTrain2D, which can be used for the following visualizations

        WriteLine($"Saving original and reconstructed images from {xTrain.Length} xTrain points and {yTrain.Length} yTrain points.");

        int[] selectedImages = [20, 21, 22, 23, 30];

        foreach (int index in selectedImages)
        {
            Utils.Drawing.SaveMnistPicture(100, index, trainingImagesForDrawing, $"{ModelName}_{bottleneckDim}_original_{index}");
            Utils.Drawing.SaveMnistPicture(100, index, yTrain, $"{ModelName}_{bottleneckDim}_reconstructed_{index}");
        }

        WriteLine();
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
