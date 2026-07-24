// Neural Networks in C♯
// File name: MeanSquaredErrorLoss4D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core.Operations;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Mean Squared Error (MSE) loss function for 4D tensors.
/// </summary>
/// <remarks>
/// <para><b>Input:</b> The predicted values (ŷ) and the target values (y) as 4D tensors.</para>
/// <para><b>Formula:</b> L = (1/n) · Σ(yᵢ - ŷᵢ)² where n depends on the reduction method</para>
/// <para><b>Gradient Formula:</b> ∂L/∂ŷᵢ = (2/n) · (ŷᵢ - yᵢ) where n depends on the reduction method</para>
/// <para><b>Description:</b> The Mean Squared Error loss for 4D tensors measures the average squared difference between predicted and target values. 
/// This variant operates on 4D tensors typically structured as [batch, channels, height, width], making it suitable for 
/// convolutional neural networks (CNNs) in tasks like image reconstruction, denoising autoencoders, super-resolution, and other pixel-wise regression problems.</para>
/// <para><b>Remarks:</b> Functionally identical to MeanSquaredErrorLoss but designed for 4D tensor operations typical in CNNs. 
/// Commonly used in autoencoding tasks where the network must reconstruct input images, or in image-to-image translation problems. 
/// The 4D implementation enables efficient batch processing of multi-channel image data. Like its 2D counterpart, 
/// it is sensitive to outliers and provides strong gradients for large errors. For perceptual quality in image generation tasks, 
/// combining MSE with perceptual losses (e.g., feature-based losses from pretrained networks) often yields better results than MSE alone. 
/// The reduction parameter controls aggregation across all four dimensions of the tensor.</para>
/// </remarks>
/// <param name="mseReduction">The reduction method applied to compute the final loss value. Default is ElementMean.</param>
public class MeanSquaredErrorLoss4D(MseReduction mseReduction = MseReduction.ElementMean) : Loss<float[,,,]>
{
    private float[,,,]? _errors;

    protected override float CalculateLoss()
        => MeanSquaredErrorLoss(Prediction, Target, out _errors, mseReduction);

    protected override float[,,,] CalculateLossGradient()
    {
        Debug.Assert(_errors != null, "_errors should not be null here.");

        return MeanSquaredErrorLossGradient(_errors, mseReduction);
    }

    public override string ToString() 
        => $"MeanSquaredError4D (mseReduction={mseReduction})";
}
