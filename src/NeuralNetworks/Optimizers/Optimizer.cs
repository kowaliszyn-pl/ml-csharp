// Neural Networks in C♯
// File name: Optimizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Runtime.InteropServices;

using NeuralNetworks.LearningRates;

namespace NeuralNetworks.Optimizers;

/// <summary>
/// Base class for a neural network optimizer.
/// </summary>
public abstract class Optimizer(LearningRate learningRate)
{
    public LearningRate LearningRate => learningRate;

    public virtual void UpdateLearningRate(int steps, int epoch, int epochs)
        => learningRate.Update(steps, epoch, epochs);

    private static void ConvertToSpans<T>(T param, T paramGradient, out Span<float> paramSpan, out ReadOnlySpan<float> paramGradientSpan) where T : notnull
    {
        paramSpan = param switch
        {
            float[] arr1D => arr1D.AsSpan(),
            float[,] arr2D => MemoryMarshal.CreateSpan(ref arr2D[0, 0], arr2D.Length),
            float[,,] arr3D => MemoryMarshal.CreateSpan(ref arr3D[0, 0, 0], arr3D.Length),
            float[,,,] arr4D => MemoryMarshal.CreateSpan(ref arr4D[0, 0, 0, 0], arr4D.Length),
            _ => throw new ArgumentException()
        };

        paramGradientSpan = paramGradient switch
        {
            float[] arr1D => arr1D.AsSpan(),
            float[,] arr2D => MemoryMarshal.CreateReadOnlySpan(ref arr2D[0, 0], arr2D.Length),
            float[,,] arr3D => MemoryMarshal.CreateReadOnlySpan(ref arr3D[0, 0, 0], arr3D.Length),
            float[,,,] arr4D => MemoryMarshal.CreateReadOnlySpan(ref arr4D[0, 0, 0, 0], arr4D.Length),
            _ => throw new ArgumentException()
        };
    }

    public void Update<T>(T paramsToUpdate, T paramGradients)
        where T : notnull
    {
        ConvertToSpans(paramsToUpdate, paramGradients, out Span<float> paramSpan, out ReadOnlySpan<float> paramGradientSpan);
        Update(paramsToUpdate, paramSpan, paramGradientSpan);
    }

    protected abstract void Update(object paramsKey, Span<float> paramsToUpdate, ReadOnlySpan<float> paramGradients);
}
