// Neural Networks in C♯
// File name: Dropout4D.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core;
using NeuralNetworks.Core.Extensions;
using NeuralNetworks.Operations.Interfaces;

namespace NeuralNetworks.Operations;

public class Dropout4D(float keepProb = 0.8f, SeededRandom? random = null) : Operation4D, IParameterCountProvider
{
    private float[,,,]? _mask;

    protected override float[,,,] CalcOutput(bool inference)
    {
        if (inference)
        {
            return Input.Multiply(keepProb);
        }
        else
        {
            _mask = Input.AsZeroOnes(keepProb, random ?? new());
            return Input.MultiplyElementwise(_mask);
        }
    }

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
    {
        Debug.Assert(_mask != null, "Mask must not be null here.");
        return outputGradient.MultiplyElementwise(_mask);
    }

    public override string ToString() => $"Dropout4D (keepProb={keepProb}, seed={random?.Seed})";

    public int GetParamCount()
        => _mask?.Length ?? 0;
}
