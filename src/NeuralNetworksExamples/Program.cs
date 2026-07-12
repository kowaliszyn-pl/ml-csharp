// Neural Networks in C♯
// File name: ProgramNew.cs
// www.kowaliszyn.pl, 2025 - 2026

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core.Operations;

using NeuralNetworksExamples.UI;

using Serilog;

using Spectre.Console;

using static System.Console;

namespace NeuralNetworksExamples;

internal static class Program
{

    internal static ILoggerFactory LoggerFactory { get; private set; } = default!;
    internal static int LatentSpaceDimensions { get; private set; } = 28;

    private static void Main()
    {
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
        OutputEncoding = System.Text.Encoding.UTF8;

        List<MenuItem> menuItems =
        [
            new("Show settings menu", ShowSettingsMenu),
            new("Dense layer models", SelectDenseLayerModel),
            new("Convolutional models (CNN)", SelectConvolutionalModel),
            new("Autoencoders", () => { }),
            new("Exit", () => running = false)
        ];

        DisplayOptions();

        while (running)
        {
            MenuItem choice = AnsiConsole.Prompt(
            new SelectionPrompt<MenuItem>()
                .Title("Select a [bold]routine[/] to run:")
                .AddChoices(menuItems)
                .UseConverter(item => item.Display));

            choice.PerformAction();

            if (choice.Display == "Exit")
            {
                AnsiConsole.MarkupLine("[yellow]Goodbye![/]");
                running = false;
            }
        }
    }

    private static void ShowSettingsMenu()
    {
        List<MenuItem> optionsMenuItems =
        [
            new("Select operation backend", SelectOperationBackend),
            new("Enter latent space dimensions", EnterLatentSpaceDimensions),
            new("Back", () => { })
        ];

        MenuItem choice = AnsiConsole.Prompt(
            new SelectionPrompt<MenuItem>()
                .Title("Select an option:")
                .AddChoices(optionsMenuItems)
                .UseConverter(item => item.Display));

        choice.PerformAction();
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
        List<MenuItem> backendMenuItems =
        [
            new("CPU - Arrays", () => OperationBackend.Use(OperationBackendType.CpuArrays)),
            new("CPU - Spans", () => OperationBackend.Use(OperationBackendType.CpuSpans)),
            new("CPU - Spans Parallel", () => OperationBackend.Use(OperationBackendType.CpuSpansParallel)),
            new("GPU", () => OperationBackend.Use(OperationBackendType.Gpu)),
            new("Back", () => { })
        ];
        MenuItem choice = AnsiConsole.Prompt(
            new SelectionPrompt<MenuItem>()
                .Title("Select an operation backend:")
                .AddChoices(backendMenuItems)
                .UseConverter(item => item.Display));

        choice.PerformAction();

        DisplayOptions();
    }

    private static void SelectDenseLayerModel()
    {
        List<MenuItem> denseLayerMenuItems =
        [
            new("Sine function approximation", SineFunction.Run, true),
            new("Boston Housing data set (custom model)", () => BostonHousing.Run(true), true),
            new("Boston Housing data set (generic model)", () => BostonHousing.Run(false), true),
            new("MNIST data set (dense layers)", MnistDense.Run, true),
            new("Load and evaluate MNIST data set (dense layers)", MnistDense.LoadAndEvaluate, true),
            new("Back", () => { })
        ];
        MenuItem choice = AnsiConsole.Prompt(
            new SelectionPrompt<MenuItem>()
                .Title("Select a dense layer model:")
                .AddChoices(denseLayerMenuItems)
                .UseConverter(item => item.Display));

        choice.PerformAction();
    }

    private static void SelectConvolutionalModel()
    {
        List<MenuItem> convolutionalMenuItems =
        [
            new("MNIST data set (CNN 2D)", MnistCnn.Run, true),
            new("ECG 200 (CNN 1D)", Ecg200.Run, true),
            new("Back", () => { })
        ];
        MenuItem choice = AnsiConsole.Prompt(
            new SelectionPrompt<MenuItem>()
                .Title("Select a convolutional model:")
                .AddChoices(convolutionalMenuItems)
                .UseConverter(item => item.Display));

        choice.PerformAction();
    }

    private static void DisplayOptions()
    {
        AnsiConsole.MarkupLine($"\nCurrent settings: backend: [green]{OperationBackend.CurrentType}[/], latent space dimensions: [green]{LatentSpaceDimensions}[/].\n");
    }
}