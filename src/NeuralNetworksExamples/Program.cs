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
            Console.WriteLine("0. Select operation backend");
            Console.WriteLine("1. Function data set");
            Console.WriteLine("2. Boston Housing data set (custom model)");
            Console.WriteLine("3. Boston Housing data set (generic model)");
            Console.WriteLine("4. MNIST data set (dense layers)");
            Console.WriteLine("5. MNIST data set (CNN)");
            Console.WriteLine("Other: Exit");
            Console.WriteLine();
            Console.Write("Enter your choice: ");

            string? choice = Console.ReadLine();
            Console.WriteLine();

            switch (choice)
            {
                case "0":
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
                case "4":
                    Mnist.Run();
                    break;
                case "5":
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
        Console.WriteLine("1. CPU - Arrays");
        Console.WriteLine("2. CPU - Spans");
        Console.WriteLine("3. CPU - Spans Parallel");
        Console.WriteLine("4. GPU");
        Console.WriteLine("Other: Exit");
        Console.WriteLine();
        Console.Write("Enter your choice: ");
        string? backendChoice = Console.ReadLine();
        Console.WriteLine();
        switch (backendChoice)
        {
            case "1":
                OperationBackend.Use(OperationBackendType.Cpu_Arrays);
                Console.WriteLine("Using CPU - Arrays backend.");
                break;
            case "2":
                OperationBackend.Use(OperationBackendType.Cpu_Spans);
                Console.WriteLine("Using CPU - Spans backend.");
                break;
            case "3":
                OperationBackend.Use(OperationBackendType.Cpu_Spans_Parallel);
                Console.WriteLine("Using CPU - Spans Parallel backend.");
                break;
            case "4":
                OperationBackend.Use(OperationBackendType.Gpu);
                Console.WriteLine("Using GPU backend.");
                break;
            default:
                Console.WriteLine("No changes made to the operation backend.");
                break;
        }
    }
}