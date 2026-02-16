// Neural Networks in C♯
// File name: Conv2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Parameterized;

/// <summary>
/// Dimensions of the input are: [batch, channels, height, width].
/// Dimensions of the param array are: [channels, filters, kernelSize, kernelSize].
/// CalcOutput returns an array of dimensions: [batch, filters, height, width].
/// Padding is assumed to be the same on all sides = kernelSize / 2
/// </summary>
/// <param name="weights"></param>
public class Conv2D(float[,,,] weights, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1) : ParamOperation<float[,,,], float[,,,], float[,,,]>(weights)
{

    protected override float[,,,] CalcOutput(bool inference)
        => Convolve2DOutput(Input, Param, paddingHeight, paddingWidth, strideHeight, strideWidth, dilatationHeight, dilatationWidth);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
        => Convolve2DInputGradient(Input, Param, outputGradient, paddingHeight, paddingWidth, strideHeight, strideWidth, dilatationHeight, dilatationWidth);

    protected override float[,,,] CalcParamGradient(float[,,,] outputGradient)
    {
        int kernelHeight = Param.GetLength(2);
        int kernelWidth = Param.GetLength(3);
        return Convolve2DParamGradient(Input, outputGradient, kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilatationHeight, dilatationWidth);
    }
}
