// Machine Learning Utils
// File name: DecayLearningRate.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.LearningRates;

public abstract class DecayLearningRate(float initialLearningRate) : LearningRate
{
    protected float CurrentLearningRate { get; set; } = initialLearningRate;

    public override float GetLearningRate() => CurrentLearningRate;
}
