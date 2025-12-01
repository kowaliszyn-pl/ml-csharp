// Machine Learning Utils
// File name: IParameterUpdater.cs
// Code It Yourself with .NET, 2024

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Operations.Interfaces;

internal interface IParameterUpdater
{
    void UpdateParams(Layer? layer, Optimizer optimizer);
}
