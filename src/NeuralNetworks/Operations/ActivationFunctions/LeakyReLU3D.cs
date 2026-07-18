// Neural Networks in C♯
// File name: LeakyReLU3D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Leaky Rectified Linear Unit (Leaky ReLU) activation function for 3D tensors.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = max(alfa · x, beta · x) = beta · x if x > 0, else alfa · x</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · beta if x > 0, else ∂L/∂y · alfa</para>
/// <para><b>Output Range:</b> (-∞, ∞)</para>
/// <para><b>Description:</b> Leaky ReLU addresses the "dying ReLU" problem by allowing a small, non-zero gradient when the input is negative. 
/// Instead of outputting zero for negative inputs, it outputs alfa · x (typically with alfa = 0.01). For positive inputs, 
/// it behaves like standard ReLU with scaling factor beta. This ensures that all neurons can continue to learn even if they receive negative inputs.</para>
/// <para><b>Remarks:</b> The typical configuration uses alfa = 0.01 and beta = 1, resulting in f(x) = x for x > 0 and f(x) = 0.01x for x ≤ 0. 
/// This small negative slope prevents neurons from dying while maintaining most of ReLU's computational benefits. 
/// Variants include Parametric ReLU (PReLU) where alfa is learned during training, and Randomized ReLU (RReLU) where alfa is randomly sampled. 
/// Leaky ReLU generally performs better than standard ReLU in practice, especially in deeper networks.</para>
/// </remarks>
/// <param name="alfa">The slope for negative inputs. Default is 0.01.</param>
/// <param name="beta">The slope for positive inputs. Default is 1.</param>
public class LeakyReLU3D(float alfa = 0.01f, float beta = 1f) : ActivationFunction<float[,,], float[,,]>
{
    protected override float[,,] CalcOutput(bool inference)
        => LeakyReLUOutput(Input, alfa, beta);

    protected override float[,,] CalcInputGradient(float[,,] outputGradient)
        => LeakyReLUInputGradient(outputGradient, Input, alfa, beta);

    public override string ToString()
        => $"LeakyReLU3D (alfa={alfa}, beta={beta})";
}
