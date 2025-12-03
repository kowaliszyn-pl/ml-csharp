// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using Microsoft.Extensions.Logging;

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
            Console.WriteLine("Select a routine to run (Neural Networks Examples):");
            Console.WriteLine("1. Function data set");
            Console.WriteLine("2. Boston Housing data set");
            Console.WriteLine("3. MNIST data set");
            Console.WriteLine("Other: Exit");
            Console.WriteLine();
            Console.Write("Enter your choice: ");

            string? choice = Console.ReadLine();
            Console.WriteLine();

            //Stopwatch stopwatch = Stopwatch.StartNew();
            switch (choice)
            {
                case "1":
                    Function.Run();
                    break;
                case "2":
                    BostonHousing.Run();
                    break;
                case "3":
                    Mnist.Run();
                    break;

                default:
                    Console.WriteLine("Goodbye!");
                    running = false;
                    break;
            }

            if (running)
            {
                //Console.ForegroundColor = ConsoleColor.Cyan;
                //Console.WriteLine($"Elapsed time: {stopwatch.Elapsed.TotalSeconds:F2} seconds.");
                //Console.ResetColor();

                Console.WriteLine("\nPress any key to continue...");
                Console.ReadKey();
                Console.WriteLine();
            }
        }
    }
}