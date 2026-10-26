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

    public void Update<T>(T paramsToUpdate, T paramGradients)
        where T : notnull
    {
        Span<float> paramsToUpdateSpan = GenericArrayRoutines.ConvertToSpan(paramsToUpdate);
        ReadOnlySpan<float> paramGradientsSpan = GenericArrayRoutines.ConvertToReadOnlySpan(paramGradients);
        Update(paramsKey: paramsToUpdate, paramsToUpdateSpan, paramGradientsSpan);
    }

    protected abstract void Update(object paramsKey, Span<float> paramsToUpdate, ReadOnlySpan<float> paramGradients);
}
