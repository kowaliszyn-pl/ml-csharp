// Machine Learning Utils
// File name: DecayLearningRate.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.LearningRates;

public abstract class DecayLearningRate(float initialLearningRate, int warmupSteps = 0) : LearningRate
{
    protected float CurrentLearningRate { get; set; } = initialLearningRate;

    protected float InitialLearningRate { get; } = initialLearningRate;

    public override float GetLearningRate() => CurrentLearningRate;

    public int WarmupSteps { get; init; } = warmupSteps;

    protected void ApplyWarmup(int steps)
    {
        if (WarmupSteps > 0 && steps < WarmupSteps) // Only for the first epoch
            CurrentLearningRate = InitialLearningRate * steps / WarmupSteps;
        else
            CurrentLearningRate = InitialLearningRate;
    }
}
