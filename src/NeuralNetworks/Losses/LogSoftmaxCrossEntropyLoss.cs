// Neural Networks in C♯
// File name: LogSoftmaxCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Categorical Cross-Entropy loss with Log-Softmax activation (also known as Negative Log-Likelihood Loss).
/// </summary>
/// <remarks>
/// <para><b>Input:</b> The predicted values (logits) and the target values (one-hot encoded) as 2D tensors.</para>
/// <para><b>Formula:</b> L = -Σᵢ yᵢ · log_softmax(ŷᵢ) = -Σᵢ yᵢ · (ŷᵢ - log(Σⱼ exp(ŷⱼ)))</para>
/// <para><b>Gradient Formula:</b> ∂L/∂ŷᵢ = softmax(ŷᵢ) - yᵢ (identical to <see cref="SoftmaxCrossEntropyLoss"/> gradient)</para>
/// <para><b>Description:</b> This loss function is mathematically equivalent to SoftmaxCrossEntropyLoss but uses a log-softmax formulation 
/// that can provide better numerical stability in some scenarios. It expects raw logits as predictions and one-hot encoded vectors as targets. 
/// Commonly used in multi-class classification tasks, particularly when numerical precision is critical or when working with very large class spaces.</para>
/// <para><b>Remarks:</b> The log-softmax formulation: log_softmax(x) = x - log(Σ exp(x)) is often more numerically stable than computing 
/// softmax followed by log, especially when dealing with very large or very small logit values. The gradient computation remains identical 
/// to <see cref="SoftmaxCrossEntropyLoss"/> (softmax_output - target), demonstrating the mathematical equivalence of the two approaches. 
/// This loss is also known as Negative Log-Likelihood (NLL) loss when used with log-softmax outputs. 
/// It's particularly useful in frameworks where log-probabilities are preferred over probabilities (e.g., for computational efficiency in certain architectures). 
/// PyTorch's CrossEntropyLoss is similar, applying log-softmax internally. The choice between this and <see cref="SoftmaxCrossEntropyLoss"/> is often a matter of 
/// implementation preference and numerical considerations rather than mathematical differences.</para>
/// </remarks>
public class LogSoftmaxCrossEntropyLoss() : Loss<float[,]>
{
    private float[,]? _softmaxOutput;

    protected override float CalculateLoss()
        => LogSoftmaxCrossEntropyLoss(Prediction, Target, out _softmaxOutput);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_softmaxOutput != null, "_softmaxOutput should not be null here.");

        // SoftmaxCrossEntropyLossGradient is valid for both SoftmaxCrossEntropyLoss and LogSoftmaxCrossEntropyLoss, as it computes the gradient of the loss with respect to the input predictions in the same way for both cases.
        return SoftmaxCrossEntropyLossGradient(_softmaxOutput, Target);
    }

    public override string ToString() => "LogSoftmaxCrossEntropyLoss";
}
