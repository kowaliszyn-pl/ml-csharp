// Neural Networks in C♯
// File name: LinearDecayLearningRate.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.LearningRates;

public class LinearDecayLearningRate(float initialLearningRate, float finalLearningRate, int warmupSteps = 0)
    : DecayLearningRate(initialLearningRate, warmupSteps)
{
    public override void Update(int steps, int epoch, int epochs)
    {
        if (epoch == 1)
            ApplyWarmup(steps);
        else
            CurrentLearningRate = InitialLearningRate - (InitialLearningRate - finalLearningRate) * (epoch - 1) / (epochs - 1);
    }

    public override string ToString()
        => $"LinearDecayLearningRate (initialLearningRate={InitialLearningRate}, finalLearningRate={finalLearningRate}, warmupSteps={WarmupSteps})";
}
