// Machine Learning Utils
// File name: SeededRandom.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.Core;

public class SeededRandom : Random
{
    public int? Seed { get; private set; }

    public SeededRandom(int seed) : base(seed)
    {
        Seed = seed;
    }

    public SeededRandom() : base()
    {
    }

    public override string ToString() => $"SeededRandom (seed={Seed})";
}
