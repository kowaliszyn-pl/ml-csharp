// Neural Networks in C♯
// File name: LinearDecayLearningRate.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.LearningRates;

public class LinearDecayLearningRate(float initialLearningRate, float finalLearningRate) : DecayLearningRate(initialLearningRate)
{
    private readonly float _initialLearningRate = initialLearningRate;
    private readonly float _finalLearningRate = finalLearningRate;

    public override void Update(int epoch, int epochs)
    {
        if (epochs == 1)
            CurrentLearningRate = _initialLearningRate;
        else
            CurrentLearningRate = _initialLearningRate - (_initialLearningRate - _finalLearningRate) * (epoch - 1) / (epochs - 1);
    }

    public override string ToString()
        => $"LinearDecayLearningRate (initialLearningRate={_initialLearningRate}, finalLearningRate={_finalLearningRate})";
}
