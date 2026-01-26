// Machine Learning Utils
// File name: ExponentialDecayLearningRate.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.LearningRates;

public class ExponentialDecayLearningRate(float initialLearningRate, float finalLearningRate, int warmupSteps = 0) : DecayLearningRate(initialLearningRate, warmupSteps)
{
    private readonly float _initialLearningRate = initialLearningRate;
    private readonly float _finalLearningRate = finalLearningRate;

    public override void Update(int steps, int epoch, int epochs)
    {
        if (WarmupSteps > 0 && steps < WarmupSteps && epoch == 1) // Only for the first epoch
        {
            CurrentLearningRate = _initialLearningRate * steps / WarmupSteps;
            return;
        }

        if (epochs == 1)
            CurrentLearningRate = _initialLearningRate;
        else
            CurrentLearningRate = _initialLearningRate * (float)Math.Pow(_finalLearningRate / _initialLearningRate, (float)(epoch - 1) / (epochs - 1));
    }

    public override string ToString() => $"ExponentialDecayLearningRate (initialLearningRate={_initialLearningRate}, finalLearningRate={_finalLearningRate}, warmupSteps={WarmupSteps})";
}
