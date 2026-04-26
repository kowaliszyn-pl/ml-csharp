// Neural Networks in C♯
// File name: MeanSquaredErrorLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Represents a loss function that computes the mean squared error between predicted and target values.
/// </summary>
/// <remarks>This class is typically used in regression tasks (for example, Boston Housing dataset, Sine wave prediction) to measure the average squared difference between
/// predicted and actual values. It can be used as a loss function in training neural networks or other predictive
/// models where minimizing the mean squared error is desired.</remarks>
public class MeanSquaredErrorLoss : Loss<float[,]>
{
    private float[,]? _errors;

    protected override float CalculateLoss()
        => MeanSquaredErrorLoss(Prediction, Target, out _errors);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_errors != null, "_errors should not be null here.");

        return MeanSquaredErrorLossGradient(_errors);
    }

    public override string ToString() => "MeanSquaredError";
}