// Neural Networks in C♯
// File name: Conv2D.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Core.ArrayUtils;
using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Parameterized;

// TODO: strides, custom padding, dilation, pooling

/// <summary>
/// Dimensions of the input are: [batch, channels, height, width].
/// Dimensions of the param array are: [channels, filters, kernelSize, kernelSize].
/// CalcOutput returns an array of dimensions: [batch, filters, height, width].
/// Padding is assumed to be the same on all sides = kernelSize / 2
/// </summary>
/// <param name="weights"></param>
public class Conv2D(float[,,,] weights) : ParamOperation<float[,,,], float[,,,], float[,,,]>(weights)
{

    protected override float[,,,] CalcOutput(bool inference) 
        => Convolve2DOutput(Input, Param);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient) 
        => Convolve2DInputGradient(Input, Param, outputGradient);

    protected override float[,,,] CalcParamGradient(float[,,,] outputGradient)
    {
        int kernelSize = Param.GetLength(2);
        return Convolve2DParamGradient(Input, outputGradient, kernelSize, kernelSize);
    }

    internal override void UpdateParams(Layer? layer, Optimizer optimizer)
        => optimizer.Update(layer, Param, ParamGradient);
}
