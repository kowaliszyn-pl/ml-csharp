// Machine Learning Utils
// File name: ConstantLearningRate.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.LearningRates;

public class ConstantLearningRate(float learningRate) : LearningRate
{
    public override float GetLearningRate() => learningRate;

    public override string ToString() => $"ConstantLearningRate (learningRate={learningRate})";
}
