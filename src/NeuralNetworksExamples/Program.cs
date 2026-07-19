// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Text.Json;

using ILGPU.Frontend;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core.Operations;

using NeuralNetworksExamples.Autoencoder;
using NeuralNetworksExamples.Cnn;
using NeuralNetworksExamples.Dense;
using NeuralNetworksExamples.UI;

using Serilog;

using Spectre.Console;

using static System.Console;

namespace NeuralNetworksExamples;

internal static class Program
{
    private const string DataFolder = "..\\..\\..\\..\\..\\data";
    internal const string MnistDataFolderPath = DataFolder + "\\MNIST";
    internal const string BostonHousingDataFilePath = DataFolder + "\\Boston\\BostonHousing.csv";
    internal const string Ecg200DataFolderPath = DataFolder + "\\ecg200";
    private const string OptionsFileName = "options.json";

    internal static ILoggerFactory LoggerFactory { get; private set; } = default!;
    internal static int LatentSpaceDimensions { get; private set; } = 28;
    internal const int MaxSamplesToVisualize = 3_500;

    private static void Main()
    {
        OutputEncoding = System.Text.Encoding.UTF8;

        AnsiConsole.Write(
           new FigletText("Neural Networks Examples")
               .Centered()
               .Color(Color.Cyan1));

        // Create ILogger using Serilog
        Serilog.Core.Logger serilog = new LoggerConfiguration()
            .WriteTo.File("..\\..\\..\\Logs\\log-.txt", rollingInterval: RollingInterval.Day)
            .CreateLogger();

        Log.Logger = serilog;
        Log.Information("Logging started...");

        // Create a LoggerFactory and add Serilog
        LoggerFactory = new LoggerFactory()
            .AddSerilog(serilog);

        bool running = true;
        
        List<MenuItem> menuItems =
        [
            new("🔚 Exit", () => running = false),
            new("🧠 Dense layer models", SelectDenseLayerModel),
            new("🖼️ Convolutional models (CNN)", SelectConvolutionalModel),
            new("🔧 Autoencoders", SelectAutoencoderModel),
            new("⚙️ Settings", ShowSettingsMenu),
        ];

        LoadOptions();

        DisplayOptions();

        while (running)
        {
            MenuItem choice = ExecuteMenu(menuItems, "Select a [bold]routine[/] to run:");

            if (choice.Display == "Exit")
            {
                AnsiConsole.MarkupLine("[yellow]Goodbye![/]");
                running = false;
            }
        }

        SaveOptions();
    }

    private static MenuItem ExecuteMenu(List<MenuItem> menuItems, string title)
    {
        MenuItem choice = AnsiConsole.Prompt(
            new SelectionPrompt<MenuItem>()
                .Title(title)
                .WrapAround()
                .HighlightStyle(new Style(decoration: Decoration.Invert))
                .AddChoices(menuItems)
                .UseConverter(item => item.Display));

        choice.PerformAction();
        return choice;
    }

    private static void ShowSettingsMenu()
    {
        ExecuteMenu([
            new("⚙️ Select operation backend", SelectOperationBackend),
            new("🧠 Enter latent space dimensions", EnterLatentSpaceDimensions),
            new("🔙 Back", () => { })
        ], "Select an option:");
    }

    private static void EnterLatentSpaceDimensions()
    {
        int dimensions = AnsiConsole.Prompt(
            new TextPrompt<int>("Enter the number of latent space dimensions:")
                .Validate(dim =>
                {
                    return dim > 0 ? ValidationResult.Success() : ValidationResult.Error("[red]Please enter a positive integer.[/]");
                }));
        LatentSpaceDimensions = dimensions;

        DisplayOptions();
    }

    private static void SelectOperationBackend()
    {
        ExecuteMenu([
            new("🖥️ CPU - Arrays", () => OperationBackend.Use(OperationBackendType.CpuArrays)),
            new("🖥️ CPU - Spans", () => OperationBackend.Use(OperationBackendType.CpuSpans)),
            new("🖥️ CPU - Spans Parallel", () => OperationBackend.Use(OperationBackendType.CpuSpansParallel)),
            new("🖥️ GPU", () => OperationBackend.Use(OperationBackendType.Gpu)),
            new("🔙 Back", () => { })
        ], "Select an [bold]operation backend[/]:");

        DisplayOptions();
    }

    private static void SelectDenseLayerModel()
    {
        ExecuteMenu([
            new("🎵 Sine function approximation", SineFunction.Run, true),
            new("🏠 Boston Housing data set (custom model)", () => BostonHousing.Run(true), true),
            new("🏠 Boston Housing data set (generic model)", () => BostonHousing.Run(false), true),
            new("🖼️ MNIST data set (dense layers)", MnistDense.Run, true),
            new("🖼️ Load and evaluate MNIST data set (dense layers)", MnistDense.LoadAndEvaluate, true),
            new("🔙 Back", () => { })
        ], "Select a [bold]dense layer[/] model:");
    }

    private static void SelectConvolutionalModel()
    {
        ExecuteMenu([
            new("🖼️ MNIST data set (CNN 2D)", MnistCnn.Run, true),
            new("📈 ECG 200 (CNN 1D)", Ecg200.Run, true),
            new("🔙 Back", () => { })
        ], "Select a [bold]convolutional[/] model:");
    }

    private static void SelectAutoencoderModel()
    {
        ExecuteMenu([
            new("🖼️ Train (dense layer autoencoder)", AutoencoderDense.Train, true),
            new("🖼️ Train (CNN autoencoder)", AutoencoderCnn.Train, true),
            new("🖼️ Load and run (dense layer autoencoder)", AutoencoderDense.Load, true),
            new("🖼️ Load and run (CNN autoencoder)", AutoencoderCnn.Load, true),
            new("📊 Visualize latent space with t-SNE (dense layer autoencoder)", AutoencoderDense.VisualizeLatentSpace, true),
            new("📊 Visualize latent space with t-SNE (CNN autoencoder)", AutoencoderCnn.VisualizeLatentSpace, true),
            new("🔙 Back", () => { })
        ], "Select [bold]MNIST autoencoder[/] operation:");
    }

    private static void DisplayOptions()
        => AnsiConsole.MarkupLine($"Current settings: backend: [green]{OperationBackend.CurrentType}[/], latent space dimensions: [green]{LatentSpaceDimensions}[/].\n");

    /// <summary>
    /// Saves options as JSON to a file.
    /// </summary>
    private static void SaveOptions()
    {
        Options options = new(OperationBackend.CurrentType.ToString(), LatentSpaceDimensions);
        string json = JsonSerializer.Serialize(options);
        File.WriteAllText(OptionsFileName, json);
    }

    private static void LoadOptions()
    {
        if (File.Exists(OptionsFileName))
        {
            string json = File.ReadAllText(OptionsFileName);
            Options? options = JsonSerializer.Deserialize<Options>(json);
            if (options is not null)
            {
                LatentSpaceDimensions = options.LatentSpaceDimensions;
                OperationBackendType backendType = Enum.Parse<OperationBackendType>(options.OperationBackendType);
                OperationBackend.Use(backendType);
            }
        }
    }
}