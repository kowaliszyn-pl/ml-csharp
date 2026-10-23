// Neural Networks in C♯
// File name: BiasAdd.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Regularization;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Parameterized;

/// <summary>
/// Computes bias addition.
/// </summary>
/// <param name="bias">Bias matrix.</param>
/// <param name="regularizer">Optional L1/L2 regularizer applied to the biases.</param>
public class BiasAdd(float[] bias, Regularizer? regularizer = null)
    : ParamOperation<float[,], float[,], float[]>(bias, regularizer)
{

    protected override float[,] CalcOutput(bool inference)
        => BiasAddOutput(Input, Param);

    protected override float[] CalcParamGradient(float[,] outputGradient)
       => BiasAddParamGradient(outputGradient);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
      => outputGradient; // Input.AsOnes().MultiplyElementwise(outputGradient);
}
