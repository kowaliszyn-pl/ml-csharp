// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core.Operations;

using NeuralNetworksExamples;

using Serilog;

using static System.Console;

internal static class Program
{
    internal static ILoggerFactory LoggerFactory { get; private set; } = default!;

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

        while (running)
        {
            bool fromSubmenu = false;
            WriteLine("Select a routine to run (Neural Networks Examples):");
            WriteLine("B. Select operation backend");
            WriteLine("S. Sine function approximation");
            WriteLine("1. Boston Housing data set (custom model)");
            WriteLine("2. Boston Housing data set (generic model)");
            WriteLine("D. MNIST data set (dense layers)");
            WriteLine("C. MNIST data set (CNN 2D)");
            WriteLine("E. ECG 200 (CNN 1D)");
            WriteLine("L. Load and evaluate MNIST data set (dense layers)");
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
                case "S":
                    SineFunction.Run();
                    break;
                case "1":
                    BostonHousing.Run(true);
                    break;
                case "2":
                    BostonHousing.Run(false);
                    break;
                case "D":
                    MnistDense.Run();
                    break;
                case "C":
                    MnistCnn.Run();
                    break;
                case "E":
                    Ecg200.Run();
                    break;
                case "L":
                    MnistDense.LoadAndEvaluate();
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
}