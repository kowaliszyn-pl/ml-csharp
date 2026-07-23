// Neural Networks in C♯
// File name: SigmoidBinaryCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Binary Cross-Entropy Loss combined with Sigmoid activation function.
/// </summary>
/// <param name="eps"></param>
public class SigmoidBinaryCrossEntropyLoss(float eps = 1e-7f) : Loss<float[,]>
{
    private float[,]? _sigmoidOutput;

    protected override float CalculateLoss()
        => SigmoidBinaryCrossEntropyLoss(Prediction, Target, out _sigmoidOutput, eps);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_sigmoidOutput != null, "_sigmoidOutput should not be null here.");

        return SigmoidBinaryCrossEntropyLossGradient(_sigmoidOutput, Target);
    }

    public override string ToString() => $"SigmoidBinaryCrossEntropyLoss (eps={eps})";
}
