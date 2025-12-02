// Machine Learning Utils
// File name: ExponentialDecayLearningRate.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.LearningRates;

public class ExponentialDecayLearningRate(float initialLearningRate, float finalLearningRate) : DecayLearningRate(initialLearningRate)
{
    private readonly float _initialLearningRate = initialLearningRate;
    private readonly float _finalLearningRate = finalLearningRate;

    public override void Update(int epoch, int epochs)
    {
        if (epochs == 1)
            CurrentLearningRate = _initialLearningRate;
        else
            CurrentLearningRate = _initialLearningRate * (float)Math.Pow(_finalLearningRate / _initialLearningRate, (float)(epoch - 1) / (epochs - 1));
        // Console.WriteLine($"CurrentLearningRate: {CurrentLearningRate}, epoch: {epoch}, epochs: {epochs}");
    }

    public override string ToString() => $"ExponentialDecayLearningRate (initialLearningRate={_initialLearningRate}, finalLearningRate={_finalLearningRate})";
}
