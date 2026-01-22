// Neural Networks in C♯
// File name: ModelUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Utils;

public static class ModelUtils
{
    public static string GetTypeIdentifier(Type type)
        => type.AssemblyQualifiedName ?? type.FullName ?? type.Name;

    public static void EnsureTypeMatch(string? persistedType, Type runtimeType, int layerIndex, int? operationIndex = null)
    {
        string expectedType = GetTypeIdentifier(runtimeType);
        if (!string.Equals(persistedType, expectedType, StringComparison.Ordinal))
        {
            string location = operationIndex is null
                ? $"Layer {layerIndex}"
                : $"Layer {layerIndex}, operation {operationIndex}";
            throw new InvalidOperationException($"Type mismatch at {location}. Expected '{expectedType}' but found '{persistedType ?? "<unknown>"}'.");
        }
    }
}
