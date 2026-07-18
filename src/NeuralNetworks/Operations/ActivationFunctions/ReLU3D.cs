// Neural Networks in C♯
// File name: ReLU3D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Rectified Linear Unit (ReLU) activation function for 3D tensors.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = max(0, beta · x)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · beta if x > 0, else 0</para>
/// <para><b>Output Range:</b> [0, ∞) when beta > 0</para>
/// <para><b>Description:</b> ReLU variant for 3D tensors, commonly used in sequence models or 3D convolutional networks. 
/// Maintains the same computational efficiency and gradient flow benefits as the standard ReLU while operating on 3D data structures 
/// such as sequences with features [batch, timesteps, features] or volumetric data.</para>
/// <para><b>Remarks:</b> Functionally identical to ReLU2D but designed for 3D tensor operations. 
/// The beta parameter (default 1) scales the positive region. Inherits the same advantages (no vanishing gradients, computational efficiency) 
/// and disadvantages (dying ReLU problem) as the standard ReLU.</para>
/// </remarks>
/// <param name="beta">The scaling factor for positive inputs. Default is 1.</param>
public class ReLU3D(float beta = 1f) : ActivationFunction<float[,,], float[,,]>
{
    protected override float[,,] CalcInputGradient(float[,,] outputGradient)
        => ReLUInputGradient(outputGradient, Input, beta);

    protected override float[,,] CalcOutput(bool inference)
        => ReLUOutput(Input, beta);

    public override string ToString()
        => $"ReLU3D (beta={beta})";
}
