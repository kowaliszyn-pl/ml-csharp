// Machine Learning Utils
// File name: DecayLearningRate.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.LearningRates;

public abstract class DecayLearningRate(float initialLearningRate, int warmupSteps = 0) : LearningRate
{
    protected float CurrentLearningRate { get; set; } = initialLearningRate;

    public override float GetLearningRate() => CurrentLearningRate;

    public int WarmupSteps { get; init; } = warmupSteps;
}
