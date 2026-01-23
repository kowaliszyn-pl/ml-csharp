// Neural Networks in C♯
// File name: ModelInputShape.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Models;

internal sealed record ModelInputShape(string InputType, int[] Dimensions)
{
    internal T CreateSyntheticSample<T>(bool firstRowOnly) where T : notnull
    {
        if (!typeof(T).IsArray)
        {
            throw new NotSupportedException($"Input type '{typeof(T)}' is not supported for shape-based initialization. Provide an initialization sample explicitly.");
        }

        Type elementType = typeof(T).GetElementType()
            ?? throw new InvalidOperationException($"Unable to determine element type for '{typeof(T)}'.");

        if (Dimensions is null || Dimensions.Length == 0)
            throw new InvalidOperationException("Persisted input shape is empty.");

        if (firstRowOnly)
            Dimensions[0] = 1;

        Array sample = Array.CreateInstance(elementType, Dimensions);
        return (T)(object)sample;
    }
}
