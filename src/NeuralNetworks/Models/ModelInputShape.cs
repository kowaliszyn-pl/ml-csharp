// Neural Networks in C♯
// File name: ModelInputShape.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Models;

internal sealed record ModelInputShape(string InputType, int[] Shape)
{
    internal T CreateSyntheticSample<T>(bool firstRowOnly) where T : notnull
    {
        if (!typeof(T).IsArray)
        {
            throw new NotSupportedException($"Input type '{typeof(T)}' is not supported for shape-based initialization. Provide an initialization sample explicitly.");
        }

        Type elementType = typeof(T).GetElementType()
            ?? throw new InvalidOperationException($"Unable to determine element type for '{typeof(T)}'.");

        if (Shape is null || Shape.Length == 0)
            throw new InvalidOperationException("Persisted input shape is empty.");

        if (firstRowOnly)
            Shape[0] = 1;

        Array sample = Array.CreateInstance(elementType, Shape);
        return (T)(object)sample;
    }
}
