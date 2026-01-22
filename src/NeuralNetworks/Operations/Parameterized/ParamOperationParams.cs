// Neural Networks in C♯
// File name: ParamOperationParams.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Operations.Parameterized;

internal sealed record ParamOperationParams(int[] Shape, float[] Values)
{
    internal static ParamOperationParams FromArray(Array array)
    {
        Type? elementType = array.GetType().GetElementType();
        if (elementType != typeof(float))
        {
            throw new NotSupportedException($"Only float parameters are supported. Got '{elementType}'.");
        }

        int[] shape = new int[array.Rank];
        for (int i = 0; i < array.Rank; i++)
        {
            shape[i] = array.GetLength(i);
        }
        float[] values = new float[array.Length];
        Buffer.BlockCopy(array, 0, values, 0, Buffer.ByteLength(array));
        return new ParamOperationParams(shape, values);
    }

    internal void CopyTo(Array destination)
    {
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