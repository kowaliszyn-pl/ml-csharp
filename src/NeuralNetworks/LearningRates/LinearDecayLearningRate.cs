// Neural Networks in C♯
// File name: LinearDecayLearningRate.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.LearningRates;

public class LinearDecayLearningRate(float initialLearningRate, float finalLearningRate, int warmupSteps = 0) : DecayLearningRate(initialLearningRate, warmupSteps)
{
    private readonly float _initialLearningRate = initialLearningRate;
    private readonly float _finalLearningRate = finalLearningRate;

    public override void Update(int steps, int epoch, int epochs)
    {
        if(WarmupSteps > 0 && steps < WarmupSteps && epoch == 1) // Only for the first epoch
        {
            CurrentLearningRate = _initialLearningRate * steps  / WarmupSteps;
            return;
        }

        if (epochs == 1)
            CurrentLearningRate = _initialLearningRate;
        else
            CurrentLearningRate = _initialLearningRate - (_initialLearningRate - _finalLearningRate) * (epoch - 1) / (epochs - 1);
    }

    public override string ToString()
        => $"LinearDecayLearningRate (initialLearningRate={_initialLearningRate}, finalLearningRate={_finalLearningRate}, warmupSteps={WarmupSteps})";
}
