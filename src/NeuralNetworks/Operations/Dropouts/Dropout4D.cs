// Neural Networks in C♯
// File name: Dropout4D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Dropouts;

public class Dropout4D(float keepProb = 0.8f, SeededRandom? random = null) : Operation4D
{
    private float[,,,]? _mask;

    protected override float[,,,] CalcOutput(bool inference)
        => DropoutOutput(Input, inference, keepProb, random, out _mask);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
    {
        Debug.Assert(_mask != null, "Mask must not be null here.");
        return DropoutInputGradient(outputGradient, _mask);
    }

    public override string ToString()
        => $"Dropout4D (keepProb={keepProb}, seed={random?.Seed})";
}
