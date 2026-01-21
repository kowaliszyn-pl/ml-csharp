// Neural Networks in C♯
// File name: ParameterDataDto.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Operations;

namespace NeuralNetworks.Layers.Dtos;

internal sealed record ParameterDataDto(int[] Shape, float[] Values)
{
    public ParameterSnapshot ToSnapshot()
    {
        int[] shapeCopy = new int[Shape.Length];
        Array.Copy(Shape, shapeCopy, Shape.Length);
        float[] valueCopy = new float[Values.Length];
        Array.Copy(Values, valueCopy, Values.Length);
        return new ParameterSnapshot(shapeCopy, valueCopy);
    }

    public static ParameterDataDto FromSnapshot(ParameterSnapshot snapshot)
    {
        int[] shapeCopy = new int[snapshot.Shape.Length];
        Array.Copy(snapshot.Shape, shapeCopy, shapeCopy.Length);
        float[] valueCopy = new float[snapshot.Values.Length];
        Array.Copy(snapshot.Values, valueCopy, valueCopy.Length);
        return new ParameterDataDto(shapeCopy, valueCopy);
    }
}
