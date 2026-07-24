// Neural Networks in C♯
// File name: SoftmaxCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Categorical Cross-Entropy loss combined with Softmax activation function.
/// </summary>
/// <remarks>
/// <para><b>Input:</b> The predicted values (logits) and the target values (one-hot encoded) as 2D tensors.</para>
/// <para><b>Formula:</b> L = -Σᵢ yᵢ · log(softmax(ŷᵢ)) = -Σᵢ yᵢ · log(exp(ŷᵢ) / Σⱼ exp(ŷⱼ))</para>
/// <para><b>Gradient Formula:</b> ∂L/∂ŷᵢ = softmax(ŷᵢ) - yᵢ (remarkably simple due to combined softmax+cross-entropy derivative). This gradient formula is the same as in <see cref="LogSoftmaxCrossEntropyLoss"/>.</para>
/// <para><b>Description:</b> This loss function combines the softmax activation with categorical cross-entropy loss, 
/// making it the standard choice for multi-class classification problems. It expects raw logits (unnormalized scores) as predictions 
/// and one-hot encoded vectors as targets. The softmax converts logits to a probability distribution, 
/// and the cross-entropy measures the dissimilarity between the predicted and target distributions.</para>
/// <para><b>Remarks:</b> Combining softmax and cross-entropy into a single operation is numerically more stable than computing them separately, 
/// as it avoids potential numerical issues with log(0) or exp(large_number). The gradient simplifies beautifully to (softmax_output - target), 
/// which is both computationally efficient and numerically stable. This loss is widely used in classification tasks like MNIST digit recognition, 
/// ImageNet classification, and any multi-class classification problem. The eps (epsilon) parameter adds numerical stability to prevent log(0). 
/// For problems with many classes, label smoothing can be applied to the targets to improve generalization. 
/// This loss naturally handles class probabilities and encourages the model to be confident in its predictions.</para>
/// </remarks>
/// <param name="eps">Small epsilon value added for numerical stability to prevent log(0). Default is 1e-7.</param>
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
