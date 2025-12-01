// Machine Learning Utils
// File name: Dropout4D.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.Exceptions;

using NeuralNetworks.Core;
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
        => outputGradient.MultiplyElementwise(_mask ?? throw new NotYetCalculatedException());

    public override string ToString() => $"Dropout4D (keepProb={keepProb}, seed={random?.Seed})";

    public int GetParamCount()
        => _mask?.Length ?? 0;
}
