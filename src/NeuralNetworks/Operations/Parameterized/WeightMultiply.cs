// Neural Networks in C♯
// File name: WeightMultiply.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Core.Operations.OperationBackend;
using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations.Parameterized;

/// <summary>
/// Weight multiplication operation for a neural network.
/// </summary>
/// <param name="weights">Weight matrix.</param>
public class WeightMultiply(float[,] weights) : ParamOperation2D<float[,]>(weights)
{
    protected override float[,] CalcOutput(bool inference)
        => WeightMultiplyOutput(Input, Param);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => WeightMultiplyInputGradient(outputGradient, Param);

    protected override float[,] CalcParamGradient(float[,] outputGradient)
        => WeightMultiplyParamGradient(Input, outputGradient);

    public override void UpdateParams(Layer? layer, Optimizer optimizer)
        => optimizer.Update(layer, Param, ParamGradient);

    protected override void EnsureSameShapeForParam(float[,]? param, float[,] paramGradient)
        => EnsureSameShape(param, paramGradient);

    public override int GetParamCount()
        => Param.Length;
}
