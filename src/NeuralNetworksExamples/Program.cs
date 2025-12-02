// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworksExamples;

bool running = true;
Console.OutputEncoding = System.Text.Encoding.UTF8;

while (running)
{
    Console.WriteLine("Select a routine to run (Boston housing dataset):");
    Console.WriteLine("1. Function data set");
    Console.WriteLine("2. Boston Housing data set");
    Console.WriteLine("3. MNIST data set");
    Console.WriteLine("Other: Exit");
    Console.WriteLine();
    Console.Write("Enter your choice: ");

    string? choice = Console.ReadLine();
    Console.WriteLine();

    Stopwatch stopwatch = Stopwatch.StartNew();
    switch (choice)
    {
        case "1":
            Function();
            break;
        case "2":
            BostonHousing();
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
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"Elapsed time: ~{stopwatch.Elapsed.TotalSeconds:F2} seconds.");
        Console.ResetColor();

        Console.WriteLine("\nPress any key to continue...");
        Console.ReadKey();
        Console.WriteLine();
    }
}

static void BostonHousing()
{
}

void Function() => throw new NotImplementedException();
