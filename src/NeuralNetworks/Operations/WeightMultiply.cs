// Neural Networks in C♯
// File name: WeightMultiply.cs
// www.kowaliszyn.pl, 2025

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations;

/// <summary>
/// Weight multiplication operation for a neural network.
/// </summary>
/// <param name="weights">Weight matrix.</param>
public class WeightMultiply(float[,] weights) : ParamOperation2D<float[,]>(weights)
{
    protected override float[,] CalcOutput(bool inference)
        => Input.MultiplyDot(Param);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => outputGradient.MultiplyDot(Param.Transpose());

    protected override float[,] CalcParamGradient(float[,] outputGradient)
        => Input.Transpose().MultiplyDot(outputGradient);

    public override void UpdateParams(Layer? layer, Optimizer optimizer) => optimizer.Update(layer, Param, ParamGradient);

    protected override void EnsureSameShapeForParam(float[,]? param, float[,] paramGradient)
        => EnsureSameShape(param, paramGradient);

    public override int GetParamCount()
        => Param.Length;
}
