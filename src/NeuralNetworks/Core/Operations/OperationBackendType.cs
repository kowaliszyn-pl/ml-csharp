// Neural Networks in C♯
// File name: OperationBackendType.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Core.Operations;

public enum OperationBackendType
{
    None,
    Cpu_Arrays,
    Cpu_Spans,
    Gpu
}
