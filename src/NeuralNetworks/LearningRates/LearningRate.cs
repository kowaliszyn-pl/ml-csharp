// Machine Learning Utils
// File name: LearningRate.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.LearningRates;

public abstract class LearningRate
{
    public abstract float GetLearningRate();

    public virtual void Update(int steps, int epoch, int epochs) { }
}
