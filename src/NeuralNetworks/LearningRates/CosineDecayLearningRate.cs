// Neural Networks in C♯
// File name: CosineDecayLearningRate.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.LearningRates;

public class CosineDecayLearningRate(float initialLearningRate, float finalLearningRate, int warmupSteps = 0)
    : DecayLearningRate(initialLearningRate, warmupSteps)
{
    public override void Update(int steps, int epoch, int epochs)
    {
        if (epoch == 1)
            ApplyWarmup(steps);
        else
            CurrentLearningRate = finalLearningRate + (InitialLearningRate - finalLearningRate) * (1 + MathF.Cos(MathF.PI * (epoch - 1) / (epochs - 1))) / 2;
    }
       
    public override string ToString()
        => $"CosineDecayLearningRate (initialLearningRate={InitialLearningRate}, finalLearningRate={finalLearningRate}, warmupSteps={WarmupSteps})";
}
