// Neural Networks in C♯
// File name: Optimizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;
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
        paramSpan = GenericArrayRoutines.ConvertToSpan(param);
        paramGradientSpan = GenericArrayRoutines.ConvertToReadOnlySpan(paramGradient);
    }

    public void Update<T>(T paramsToUpdate, T paramGradients)
        where T : notnull
    {
        ConvertToSpans(paramsToUpdate, paramGradients, out Span<float> paramSpan, out ReadOnlySpan<float> paramGradientSpan);
        Update(paramsToUpdate, paramSpan, paramGradientSpan);
    }

    protected abstract void Update(object paramsKey, Span<float> paramsToUpdate, ReadOnlySpan<float> paramGradients);
}
