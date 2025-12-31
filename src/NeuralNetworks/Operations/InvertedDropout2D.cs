// Neural Networks in C♯
// File name: InvertedDropout2D.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core;
using NeuralNetworks.Core.Extensions;

namespace NeuralNetworks.Operations;

public class InvertedDropout2D(float keepProb = 0.8f, SeededRandom? random = null) : BaseDropout2D
{
    readonly float _multiplier = 1f / keepProb;

    protected override float[,] CalcOutput(bool inference)
    {
        if (inference)
        {
            return Input;
        }
        else
        {
            Mask = Input.AsZeroOnes(keepProb, random ?? new());
            return Input.MultiplyElementwise(Mask).Multiply(_multiplier);
        }
    }

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        Debug.Assert(Mask != null, "Mask must not be null here.");

        return outputGradient.MultiplyElementwise(Mask).Multiply(_multiplier);
    }

    public override string ToString() => $"InvertedDropout2D (keepProb={keepProb}, seed={random?.Seed})";
}
