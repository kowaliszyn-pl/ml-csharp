// Neural Networks in C♯
// File name: BiasAddConv1D.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Operations.Parameterized;

public class BiasAddConv1D(float[] bias) : ParamOperation<float[,,], float[,,], float[]>(bias)
{
    protected override float[,,] CalcInputGradient(float[,,] outputGradient) => throw new NotImplementedException();
    protected override float[,,] CalcOutput(bool inference) => throw new NotImplementedException();
    protected override float[] CalcParamGradient(float[,,] outputGradient) => throw new NotImplementedException();
    internal override void UpdateParams(Layer? layer, Optimizer optimizer) => throw new NotImplementedException();
}
