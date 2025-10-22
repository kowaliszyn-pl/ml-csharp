// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

Console.WriteLine("Select a routine to run:");
Console.WriteLine("1. Variables");
Console.WriteLine("2. Tables");
Console.WriteLine("3. Matrices");
Console.WriteLine("4. Matrices with bias");
// exit
Console.WriteLine("Other: Exit");
Console.Write("Enter your choice: ");

string? choice = Console.ReadLine();

switch (choice)
{
    case "1":
        // Variables();
        break;
    case "2":
        // Tables();
        break;
    case "3":
        // Matrices();
        break;
    case "4":
        // MatricesWithBias();
        break;
    default:
        Console.WriteLine("Goodbye!");
        break;
}
