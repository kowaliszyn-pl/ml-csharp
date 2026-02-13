// Neural Networks in C♯
// File name: Dropout4D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Dropouts;

public class Dropout4D(float keepProb = 0.8f, SeededRandom? random = null) : BaseDropout<float[,,,]>
{
    protected override float[,,,] CalcOutput(bool inference)
    {
        float[,,,] result = DropoutOutput(Input, inference, keepProb, random, out float[,,,]? mask);
        Mask = mask;
        return result;
    }

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
    {
        Debug.Assert(Mask != null, "Mask must not be null here.");
        return DropoutInputGradient(outputGradient, Mask);
    }

    public override string ToString()
        => $"Dropout4D (keepProb={keepProb}, seed={random?.Seed})";
}
