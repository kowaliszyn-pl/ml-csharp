// Neural Networks in C♯
// File name: InvertedDropout2D.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core;
using NeuralNetworks.Operations.Interfaces;

namespace NeuralNetworks.Operations;

public class InvertedDropout2D(float keepProb = 0.8f, SeededRandom? random = null) : Operation2D, IParameterCountProvider
{
    private float[,]? _mask;
    readonly float _multiplier = 1f / keepProb;

    protected override float[,] CalcOutput(bool inference)
    {
        if (inference)
        {
            return Input;
        }
        else
        {
            _mask = Input.AsZeroOnes(keepProb, random ?? new());
            return Input.MultiplyElementwise(_mask).Multiply(_multiplier);
        }
    }

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        Debug.Assert(_mask != null, "Mask must not be null here.");
        return outputGradient.MultiplyElementwise(_mask).Multiply(_multiplier);
    }

    public override string ToString() => $"InvertedDropout2D (keepProb={keepProb}, seed={random?.Seed})";

    public int GetParamCount()
        => _mask?.Length ?? 0;

}
