// Neural Networks in C♯
// File name: Dropout2D.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core;
using NeuralNetworks.Core.Extensions;

namespace NeuralNetworks.Operations;

public class Dropout2D(float keepProb = 0.8f, SeededRandom? random = null) : BaseDropout2D
{
    protected override float[,] CalcOutput(bool inference)
    {
        if (inference)
        {
            return Input.Multiply(keepProb);
        }
        else
        {
            Mask = Input.AsZeroOnes(keepProb, random ?? new());
            return Input.MultiplyElementwise(Mask);
        }
    }

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        Debug.Assert(Mask != null, "Mask must not be null here.");

        return outputGradient.MultiplyElementwise(Mask);
    }

    public override string ToString() 
        => $"Dropout2D (keepProb={keepProb}, seed={random?.Seed})";



}
