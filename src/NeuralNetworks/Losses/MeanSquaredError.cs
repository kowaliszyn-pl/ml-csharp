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
        return Prediction.Subtract(Target).Power(2).Sum() / batchSize;
    }

    protected override float[,] CalculateLossGradient()
    {
        int batchSize = Prediction.GetLength(0);
        return Prediction.Subtract(Target).Multiply(2f / batchSize);
    }
}
/*
{
    protected override float CalculateLoss()
    {
        int batchSize = Prediction.GetDimension(Dimension.Rows);
        // The quadratic function has the property that values further from the minimum have a steeper gradient.
        return Prediction.Subtract(Target).Power(2).Sum() / batchSize;
    }

    protected override Matrix CalculateLossGradient()
    {
        int batchSize = Prediction.GetDimension(Dimension.Rows);
        return Prediction.Subtract(Target).Multiply(2f / batchSize);
    }
}
*/