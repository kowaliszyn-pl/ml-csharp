// Neural Networks in C♯
// File name: Conv2D.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Core.ArrayUtils;
using static NeuralNetworks.Core.Span.OperationOps;

namespace NeuralNetworks.Operations;

// TODO: strides, custom padding, dilation

/// <summary>
/// Dimensions of the input are: [batch, channels, height, width]
/// Dimensions of the param array are: [channels, filters, kernelSize, kernelSize]
/// Padding is assumed to be the same on all sides = kernelSize / 2
/// </summary>
/// <param name="weights"></param>
public class Conv2D(float[,,,] weights) : ParamOperation4D<float[,,,]>(weights)
{

    protected override float[,,,] CalcOutput(bool inference) 
        => Convolve2DForward(Input, Param);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient) 
        => Convolve2DBackwardInput(Input, Param, outputGradient);

    protected override float[,,,] CalcParamGradient(float[,,,] outputGradient)
    {
        int kernelSize = Param.GetLength(2);
        return Convolve2DBackwardWeights(Input, outputGradient, kernelSize, kernelSize);
    }

    public override void UpdateParams(Layer? layer, Optimizer optimizer)
        => optimizer.Update(layer, Param, ParamGradient);

    protected override void EnsureSameShapeForParam(float[,,,]? param, float[,,,] paramGradient)
        => EnsureSameShape(param, paramGradient);

    public override int GetParamCount()
        => Param.Length;
}
