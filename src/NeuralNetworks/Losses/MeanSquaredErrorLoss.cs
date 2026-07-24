// Neural Networks in C♯
// File name: MeanSquaredErrorLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core.Operations;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Mean Squared Error (MSE) loss function for 2D tensors.
/// </summary>
/// <remarks>
/// <para><b>Input:</b> The predicted values (ŷ) and the target values (y) as 2D tensors.</para>
/// <para><b>Formula:</b> L = (1/n) · Σ(yᵢ - ŷᵢ)² where n depends on the reduction method</para>
/// <para><b>Gradient Formula:</b> ∂L/∂ŷᵢ = (2/n) · (ŷᵢ - yᵢ) where n depends on the reduction method</para>
/// <para><b>Description:</b> The Mean Squared Error loss measures the average squared difference between predicted values (ŷ) and target values (y). 
/// It is the most commonly used loss function for regression tasks, where the goal is to predict continuous values. 
/// MSE heavily penalizes large errors due to the squaring operation, making it sensitive to outliers.</para>
/// <para><b>Remarks:</b> MSE is differentiable everywhere and provides strong gradients for large errors, which can accelerate learning. 
/// However, this same property makes it sensitive to outliers - a single large error can dominate the loss value. 
/// The reduction parameter controls how the error is aggregated: ElementMean averages over all elements, 
/// while other reductions might sum or average differently. MSE is equivalent to L2 loss and is closely related to Euclidean distance. 
/// Common applications include Boston Housing price prediction, sine wave approximation, and other regression problems. 
/// For tasks with significant outliers, consider Huber loss or Mean Absolute Error (MAE) as alternatives.</para>
/// </remarks>
/// <param name="mseReduction">The reduction method applied to compute the final loss value. Default is ElementMean.</param>
public class MeanSquaredErrorLoss(MseReduction mseReduction = MseReduction.ElementMean) : Loss<float[,]>
{
    private float[,]? _errors;

    protected override float CalculateLoss()
        => MeanSquaredErrorLoss(Prediction, Target, out _errors, mseReduction);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_errors != null, "_errors should not be null here.");

        return MeanSquaredErrorLossGradient(_errors, mseReduction);
    }

    public override string ToString() => $"MeanSquaredError (mseReduction={mseReduction})";
}