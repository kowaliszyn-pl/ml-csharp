// Neural Networks in C♯
// File name: Softplus.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Softplus activation function.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = log(1 + exp(x)) = ln(1 + e^x)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · exp(x)/(1 + exp(x)) = ∂L/∂y · σ(x)</para>
/// <para><b>Output Range:</b> (0, ∞)</para>
/// <para><b>Description:</b> Softplus is a smooth, differentiable approximation of the <see cref="ReLU2D"/> function. 
/// Unlike ReLU, which has a sharp corner at zero, Softplus is continuously differentiable everywhere. 
/// It asymptotically approaches zero for large negative values and linearly increases for large positive values (f(x) ≈ x when x >> 0). 
/// Often used when a smooth activation is preferred over the non-differentiability of ReLU.</para>
/// <para><b>Remarks:</b> The derivative of Softplus is the sigmoid function: f'(x) = σ(x). 
/// For large positive x, Softplus approximates ReLU closely (f(x) ≈ x), but for x near zero, it provides smooth transitions. 
/// This can be beneficial for optimization but comes at the cost of computational expense due to the exponential and logarithm operations. 
/// Softplus is sometimes used in variational autoencoders (VAEs) and other probabilistic models. 
/// Note that for very large x, numerical stability must be considered to avoid overflow in exp(x).</para>
/// </remarks>
public class Softplus : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => SoftplusInputGradient(outputGradient, Output);

    protected override float[,] CalcOutput(bool inference)
        => SoftplusOutput(Input);

    public override string ToString() 
        => "Softplus";
}
