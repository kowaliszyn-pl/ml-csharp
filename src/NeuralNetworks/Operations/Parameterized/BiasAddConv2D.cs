// Neural Networks in C♯
// File name: BiasAddConv1D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Parameterized;

public class BiasAddConv2D(float[] bias) : ParamOperation<float[,,,], float[,,,], float[]>(bias)
{
    protected override float[,,,] CalcOutput(bool inference)
        => BiasAddConv2DOutput(Input, Param);

    protected override float[] CalcParamGradient(float[,,,] outputGradient)
        => BiasAddConv2DParamGradient(outputGradient);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
        // The input gradient for a bias addition operation is simply the output gradient, since the bias does not affect the input.
        // Therefore, the function returns the output gradient as the input gradient without modification.
        => outputGradient;
}
