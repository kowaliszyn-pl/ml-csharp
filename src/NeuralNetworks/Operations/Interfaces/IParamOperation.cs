// Neural Networks in C♯
// File name: IParamOperation.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Operations.Interfaces;

internal interface IParamOperation
{
    public int GetParamCount();
    public void UpdateParams(Layer? layer, Optimizer optimizer);
    public ParameterSnapshot Capture();
    public void Restore(ParameterSnapshot snapshot);
}
