// Neural Networks in C♯
// File name: Conv1D.cs
// www.kowaliszyn.pl, 2025 - 2026
/*
using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Core.ArrayUtils;
using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Parameterized;

public class Conv1D(float[,,] weights) : ParamOperation3D<float[,,]>(weights)
{
    protected override float[,,] CalcOutput(bool inference)
        => Convolve1DOutput(Input, Param);

    protected override float[,,] CalcInputGradient(float[,,] outputGradient)
        => Convolve1DInputGradient(Input, Param, outputGradient);

    protected override float[,,] CalcParamGradient(float[,,] outputGradient)
    {
        int kernelSize = Param.GetLength(2);
        return Convolve1DParamGradient(Input, outputGradient, kernelSize, kernelSize);
    }

    internal override void UpdateParams(Layer? layer, Optimizer optimizer)
        => optimizer.Update(layer, Param, ParamGradient);

    protected override void EnsureSameShapeForParam(float[,,]? param, float[,,] paramGradient)
        => EnsureSameShape(param, paramGradient);

    internal override int GetParamCount()
        => Param.Length;
}
*/