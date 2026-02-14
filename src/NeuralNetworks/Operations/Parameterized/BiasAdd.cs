// Neural Networks in C♯
// File name: BiasAdd.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Parameterized;

/// <summary>
/// Computes bias addition.
/// </summary>
/// <param name="bias">Bias matrix.</param>
public class BiasAdd(float[] bias) : ParamOperation<float[,], float[,], float[]>(bias)
{

    protected override float[,] CalcOutput(bool inference)
        => BiasAddOutput(Input, Param);

    protected override float[] CalcParamGradient(float[,] outputGradient)
       => BiasAddParamGradient(outputGradient);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
      => outputGradient; // Input.AsOnes().MultiplyElementwise(outputGradient);
}
