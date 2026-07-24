// Neural Networks in C♯
// File name: CrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Categorical Cross-Entropy Loss without Softmax activation function.
/// </summary>
/// <remarks>
/// Usually used in classification problems where the target is a one-hot encoded vector and the prediction is a probability distribution over classes.
/// This class uses the <see cref="Core.Operations.OperationBackend.SoftmaxCrossEntropyLossGradient"/> method to compute the loss gradient, which is valid for both <see cref="SoftmaxCrossEntropyLoss"/> and <see cref="CrossEntropyLoss"/>.
/// </remarks>
public class CrossEntropyLoss() : Loss<float[,]>
{
    private float[,]? _softmaxOutput;

    protected override float CalculateLoss()
        => CrossEntropyLoss(Prediction, Target, out _softmaxOutput);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_softmaxOutput != null, "_softmaxOutput should not be null here.");

        // SoftmaxCrossEntropyLossGradient is valid for both SoftmaxCrossEntropyLoss and CrossEntropyLoss, as it computes the gradient of the loss with respect to the input predictions in the same way for both cases.
        return SoftmaxCrossEntropyLossGradient(_softmaxOutput, Target);
    }

    public override string ToString() => $"CrossEntropyLoss";   
}
