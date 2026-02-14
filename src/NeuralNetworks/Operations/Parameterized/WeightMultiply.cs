// Neural Networks in C♯
// File name: WeightMultiply.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Parameterized;

/// <summary>
/// Weight multiplication operation for a neural network.
/// </summary>
/// <param name="weights">Weight matrix.</param>
public class WeightMultiply(float[,] weights) : ParamOperation<float[,], float[,], float[,]>(weights)
{
    protected override float[,] CalcOutput(bool inference)
        => WeightMultiplyOutput(Input, Param);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => WeightMultiplyInputGradient(outputGradient, Param);

    protected override float[,] CalcParamGradient(float[,] outputGradient)
        => WeightMultiplyParamGradient(Input, outputGradient);
}
