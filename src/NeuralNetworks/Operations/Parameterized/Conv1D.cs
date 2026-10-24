// Neural Networks in C♯
// File name: Conv1D.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Regularization;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Parameterized;

public class Conv1D(float[,,] weights, int padding, int stride = 1, int dilatation = 1, IRegularizer? regularizer = null) : ParamOperation<float[,,], float[,,], float[,,]>(weights, regularizer)
{
    protected override float[,,] CalcOutput(bool inference)
        => Convolve1DOutput(Input, Param, padding, stride, dilatation);

    protected override float[,,] CalcInputGradient(float[,,] outputGradient)
        => Convolve1DInputGradient(Input, Param, outputGradient, padding, stride, dilatation);

    protected override float[,,] CalcParamGradient(float[,,] outputGradient)
    {
        int kernelLength = Param.GetLength(2);
        return Convolve1DParamGradient(Input, outputGradient, kernelLength, padding, stride, dilatation);
    }

}
