// Neural Networks in C♯
// File name: IParameterStateProvider.cs
// www.kowaliszyn.pl, 2026

using NeuralNetworks.Operations;

namespace NeuralNetworks.Operations.Interfaces;

internal interface IParameterStateProvider
{
    ParameterSnapshot Capture();
    void Restore(ParameterSnapshot snapshot);
}
