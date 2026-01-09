// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core.Operations;

using NeuralNetworksExamples;

using Serilog;

internal static class Program
{
    internal static ILoggerFactory LoggerFactory { get; private set; } = default!;

    private static void Main(string[] args)
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
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        while (running)
        {
            bool fromSubmenu = false;
            Console.WriteLine("Select a routine to run (Neural Networks Examples):");
            Console.WriteLine("B. Select operation backend");
            Console.WriteLine("1. Function data set");
            Console.WriteLine("2. Boston Housing data set (custom model)");
            Console.WriteLine("3. Boston Housing data set (generic model)");
            Console.WriteLine("D. MNIST data set (dense layers)");
            Console.WriteLine("C. MNIST data set (CNN)");
            Console.WriteLine("Other: Exit");
            Console.WriteLine();
            Console.Write("Enter your choice: ");

            string? choice = Console.ReadLine();
            Console.WriteLine();

            switch (choice?.ToUpper())
            {
                case "B":
                    SelectOperationBackend();
                    Console.WriteLine();
                    fromSubmenu = true;
                    break;
                case "1":
                    Function.Run();
                    break;
                case "2":
                    BostonHousing.Run(true);
                    break;
                case "3":
                    BostonHousing.Run(false);
                    break;
                case "D":
                    MnistDense.Run();
                    break;
                case "C":
                    MnistCnn.Run();
                    break;

                default:
                    Console.WriteLine("Goodbye!");
                    running = false;
                    break;
            }

            if (running && !fromSubmenu)
            {
                Console.WriteLine("\nPress any key to continue...");
                Console.ReadKey();
                Console.WriteLine();
            }
        }
    }

    private static void SelectOperationBackend()
    {
        Console.WriteLine("Select operation backend:");
        Console.WriteLine("A. CPU - Arrays");
        Console.WriteLine("S. CPU - Spans");
        Console.WriteLine("P. CPU - Spans Parallel");
        Console.WriteLine("G. GPU");
        Console.WriteLine("Other: Exit");
        Console.WriteLine();
        Console.Write("Enter your choice: ");
        string? backendChoice = Console.ReadLine();
        Console.WriteLine();
        switch (backendChoice?.ToUpper())
        {
            case "A":
                OperationBackend.Use(OperationBackendType.CpuArrays);
                Console.WriteLine("Using CPU - Arrays backend.");
                break;
            case "S":
                OperationBackend.Use(OperationBackendType.CpuSpans);
                Console.WriteLine("Using CPU - Spans backend.");
                break;
            case "P":
                OperationBackend.Use(OperationBackendType.CpuSpansParallel);
                Console.WriteLine("Using CPU - Spans Parallel backend.");
                break;
            case "G":
                OperationBackend.Use(OperationBackendType.Gpu);
                Console.WriteLine("Using GPU backend.");
                break;
            default:
                Console.WriteLine("No changes made to the operation backend.");
                break;
        }
    }
}