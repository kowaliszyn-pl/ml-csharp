// Neural Networks in C♯
// File name: SoftmaxCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Categorical Cross-Entropy Loss combined with Softmax activation function.
/// </summary>
/// <param name="eps"></param>
/// <remarks>
/// Usually used in MNIST-like classification problems, where the target is a one-hot encoded vector and the prediction is a probability distribution over classes.
/// This class uses the <see cref="Core.Operations.OperationBackend.SoftmaxCrossEntropyLossGradient"/> method to compute the loss gradient, which is valid for both <see cref="SoftmaxCrossEntropyLoss"/> and <see cref="LogSoftmaxCrossEntropyLoss"/>.
/// </remarks>
public class SoftmaxCrossEntropyLoss(float eps = 1e-7f) : Loss<float[,]>
{
    private float[,]? _softmaxOutput;

    protected override float CalculateLoss() 
        => SoftmaxCrossEntropyLoss(Prediction, Target, out _softmaxOutput, eps);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_softmaxOutput != null, "_softmaxPrediction should not be null here.");

        // SoftmaxCrossEntropyLossGradient is valid for both SoftmaxCrossEntropyLoss and CrossEntropyLoss, as it computes the gradient of the loss with respect to the input predictions in the same way for both cases.
        return SoftmaxCrossEntropyLossGradient(_softmaxOutput, Target);
    }

    public override string ToString() => $"SoftmaxCrossEntropyLoss (eps={eps})";
}
