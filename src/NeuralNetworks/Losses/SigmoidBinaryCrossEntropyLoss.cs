// Neural Networks in C♯
// File name: SigmoidBinaryCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Binary Cross-Entropy loss combined with Sigmoid activation function.
/// </summary>
/// <remarks>
/// <para><b>Input:</b> The predicted values (logits) and the target values (binary labels) as 2D tensors.</para>
/// <para><b>Formula:</b> L = -[y · log(σ(ŷ)) + (1 - y) · log(1 - σ(ŷ))] where σ(ŷ) = 1/(1 + e^(-ŷ))</para>
/// <para><b>Gradient Formula:</b> ∂L/∂ŷᵢ = σ(ŷᵢ) - yᵢ (elegantly simple due to combined sigmoid+BCE derivative)</para>
/// <para><b>Description:</b> This loss function combines the sigmoid activation with binary cross-entropy loss, 
/// making it the standard choice for binary classification problems. It expects raw logits (unnormalized scores) as predictions 
/// and binary labels (0 or 1) as targets. The sigmoid converts logits to probabilities in the range (0, 1), 
/// and the binary cross-entropy measures how well the predicted probabilities match the true binary labels.</para>
/// <para><b>Remarks:</b> Combining sigmoid and binary cross-entropy into a single operation provides numerical stability 
/// by avoiding potential issues with log(0) or extreme sigmoid values. The gradient simplifies to (sigmoid_output - target), 
/// which is computationally efficient and mirrors the elegant gradient of softmax cross-entropy for multi-class problems. 
/// This loss is widely used in binary classification tasks such as spam detection, medical diagnosis (disease/no disease), 
/// sentiment analysis (positive/negative), and any two-class classification problem. The eps (epsilon) parameter adds numerical stability. 
/// For multi-label classification (where each sample can belong to multiple classes simultaneously), this loss can be applied independently to each label. 
/// Unlike multi-class cross-entropy which uses one-hot encoding, binary cross-entropy works directly with scalar 0/1 targets. 
/// This loss naturally handles probability outputs and encourages confident predictions toward 0 or 1.</para>
/// </remarks>
/// <param name="eps">Small epsilon value added for numerical stability to prevent log(0) and log(1). Default is 1e-7.</param>
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
