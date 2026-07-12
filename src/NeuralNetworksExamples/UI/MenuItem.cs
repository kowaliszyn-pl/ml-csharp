// Neural Networks in C♯
// File name: MenuItem.cs
// www.kowaliszyn.pl, 2025 - 2026

using Spectre.Console;

namespace NeuralNetworksExamples.UI;

internal record MenuItem(string Display, Action Action, bool WaitForAnyKey = false)
{
    public void PerformAction()
    {
        Action();
        if (WaitForAnyKey)
        {
            AnsiConsole.MarkupLine("\n[grey]Press any key to continue...[/]\n");
            Console.ReadKey(true);
        }
    }
}
