// Neural Networks in C♯
// File name: MeanSquaredErrorLoss4D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <remarks>
/// Usually used in convolutional neural networks (CNNs) for autoencoding tasks, where the input and output are 4D tensors (e.g., batch size, channels, height, width).
/// </remarks>
public class MeanSquaredErrorLoss4D : Loss<float[,,,]>
{
    private float[,,,]? _errors;

    protected override float CalculateLoss()
        => MeanSquaredErrorLoss(Prediction, Target, out _errors);

    protected override float[,,,] CalculateLossGradient()
    {
        Debug.Assert(_errors != null, "_errors should not be null here.");

        return MeanSquaredErrorLossGradient(_errors);
    }

    public override string ToString() => "MeanSquaredError4D";
}
