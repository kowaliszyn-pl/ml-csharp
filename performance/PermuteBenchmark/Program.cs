// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

using BenchmarkDotNet.Running;

namespace PermuteBenchmark;

internal class Program
{
    static void Main(string[] args)
    {
        var _ = BenchmarkRunner.Run(typeof(Program).Assembly);
    }
}
