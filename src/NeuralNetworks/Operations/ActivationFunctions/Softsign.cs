// Neural Networks in C♯
// File name: Softsign.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Softsign activation function.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = x / (1 + |x|)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · 1 / (1 + |x|)²</para>
/// <para><b>Output Range:</b> (-1, 1)</para>
/// <para><b>Description:</b> Softsign is a smooth activation function similar to <see cref="Tanh2D"/> but with a different rate of convergence to its asymptotes. 
/// It maps inputs to the range (-1, 1) and is defined as x divided by (1 + |x|). 
/// Softsign approaches its limits more gradually than tanh, which can lead to better gradient flow in some scenarios.</para>
/// <para><b>Remarks:</b> Compared to tanh, Softsign converges polynomially to its asymptotes (±1), while tanh converges exponentially. 
/// This means Softsign has a gentler slope near the extremes, potentially mitigating vanishing gradient issues better than tanh. 
/// However, it is computationally more expensive than ReLU and less commonly used in modern architectures. 
/// The absolute value operation makes it non-differentiable at x = 0, though in practice the gradient is typically defined as 1 at this point. 
/// Softsign can be useful in recurrent networks or when a bounded, smooth activation is desired.</para>
/// </remarks>
public class Softsign : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcOutput(bool inference)
        => SoftsignOutput(Input);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => SoftsignInputGradient(outputGradient, Input);

    public override string ToString()
        => "Softsign";
}
