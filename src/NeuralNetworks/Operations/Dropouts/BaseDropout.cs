// Neural Networks in C♯
// File name: Dropout.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Operations.Dropouts;

public abstract class BaseDropout<T> : Operation<T, T>
    where T : notnull
{
    protected T? Mask { get; set; }
}
