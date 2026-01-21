// Neural Networks in C♯
// File name: ParameterSnapshot.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Operations;

public readonly record struct ParameterSnapshot
{
    public ParameterSnapshot(int[] shape, float[] values)
    {
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        Values = values ?? throw new ArgumentNullException(nameof(values));
    }

    public int[] Shape { get; }

    public float[] Values { get; }

    public static ParameterSnapshot FromArray(Array source)
    {
        ArgumentNullException.ThrowIfNull(source);

        Type? elementType = source.GetType().GetElementType();
        if (elementType != typeof(float))
        {
            throw new NotSupportedException($"Only float parameters are supported. Got '{elementType}'.");
        }

        int[] shape = new int[source.Rank];
        for (int dim = 0; dim < source.Rank; dim++)
        {
            shape[dim] = source.GetLength(dim);
        }

        float[] values = new float[source.Length];
        Buffer.BlockCopy(source, 0, values, 0, values.Length * sizeof(float));

        return new ParameterSnapshot(shape, values);
    }

    public void CopyTo(Array destination)
    {
        ArgumentNullException.ThrowIfNull(destination);

        Type? elementType = destination.GetType().GetElementType();
        if (elementType != typeof(float))
        {
            throw new NotSupportedException($"Only float parameters are supported. Got '{elementType}'.");
        }

        if (destination.Length != Values.Length)
        {
            throw new InvalidOperationException($"Parameter size mismatch. Expected {destination.Length} values, but snapshot contains {Values.Length}.");
        }

        Buffer.BlockCopy(Values, 0, destination, 0, Values.Length * sizeof(float));
    }
}
