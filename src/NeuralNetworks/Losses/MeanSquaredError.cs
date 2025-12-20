// Neural Networks in C♯
// File name: MeanSquaredError.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;

namespace NeuralNetworks.Losses;

public class MeanSquaredError : Loss2D
{
    protected override float CalculateLoss()
    {
        int batchSize = Prediction.GetLength(0);
        // The quadratic function has the property that values further from the minimum have a steeper gradient.
        return Target.Subtract(Prediction).Power(2).Sum() / batchSize;
    }

    protected override float[,] CalculateLossGradient()
    {
        int batchSize = Prediction.GetLength(0);
        return Target.Subtract(Prediction).Multiply(-2f / batchSize);
    }

    override public string ToString() => "MeanSquaredError";
}