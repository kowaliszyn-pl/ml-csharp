// Neural Networks in C♯
// File name: WeightMultiply.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Regularization;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Parameterized;

/// <summary>
/// Weight multiplication operation for a neural network.
/// </summary>
/// <param name="weights">Weight matrix.</param>
/// <param name="regularizer">Optional regularizer applied to the weights.</param>
public class WeightMultiply(float[,] weights, IRegularizer? regularizer = null) : ParamOperation<float[,], float[,], float[,]>(weights, regularizer)
{
    protected override float[,] CalcOutput(bool inference)
        => WeightMultiplyOutput(Input, Param);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => WeightMultiplyInputGradient(outputGradient, Param);

    protected override float[,] CalcParamGradient(float[,] outputGradient)
        => WeightMultiplyParamGradient(Input, outputGradient);
}
