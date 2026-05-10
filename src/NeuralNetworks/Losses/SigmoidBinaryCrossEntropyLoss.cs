// Neural Networks in C♯
// File name: SigmoidBinaryCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

public class SigmoidBinaryCrossEntropyLoss(float eps = 1e-7f) : Loss<float[,]>
{
    private float[,]? _sigmoidOutput;

    protected override float CalculateLoss()
        // Calculate Sigmoid inside the loss function just like Softmax in SoftmaxCrossEntropyLoss, to make the gradient easier to calculate and more numerically stable. Apply Sigmoid to logits in the eval function (like the Argmax in MNIST evaluation).

        => SigmoidBinaryCrossEntropyLoss(Prediction, Target, out _sigmoidOutput, eps);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_sigmoidOutput != null, "_sigmoidOutput should not be null here.");

        return SigmoidBinaryCrossEntropyLossGradient(_sigmoidOutput, Target);
    }

    public override string ToString() => $"SigmoidBinaryCrossEntropyLoss (eps={eps})";
}
