// Machine Learning Utils
// File name: MeanSquaredError.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

namespace NeuralNetworks.Losses;

public class MeanSquaredError : Loss2D
{
    protected override float CalculateLoss()
    {
        int batchSize = Prediction.GetLength((int)Dimension.Rows);
        // The quadratic function has the property that values further from the minimum have a steeper gradient.
        return Prediction.Subtract(Target).Power(2).Sum() / batchSize;
    }

    protected override float[,] CalculateLossGradient()
    {
        int batchSize = Prediction.GetLength((int)Dimension.Rows);
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