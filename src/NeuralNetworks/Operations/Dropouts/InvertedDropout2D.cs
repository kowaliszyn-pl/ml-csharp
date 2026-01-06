// Neural Networks in C♯
// File name: InvertedDropout2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Dropouts;

public class InvertedDropout2D(float keepProb = 0.8f, SeededRandom? random = null) : BaseDropout2D
{
    protected override float[,] CalcOutput(bool inference)
    {
        float[,] result = InvertedDropoutOutput(Input, inference, keepProb, random, out float[,]? mask);
        Mask = mask;
        return result;
    }

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        Debug.Assert(Mask != null, "Mask must not be null here.");
        return InvertedDropoutInputGradient(outputGradient, Mask, keepProb);
    }

    public override string ToString()
        => $"InvertedDropout2D (keepProb={keepProb}, seed={random?.Seed})";
}
