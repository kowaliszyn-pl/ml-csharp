// Neural Networks in C♯
// File name: ReLU.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Rectified Linear Unit (ReLU) activation function for 2D tensors.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = max(0, beta · x)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · beta if x > 0, else 0</para>
/// <para><b>Output Range:</b> [0, ∞) when beta > 0</para>
/// <para><b>Description:</b> ReLU is one of the most widely used activation functions in deep learning. 
/// It outputs zero for negative inputs and scales positive inputs by the beta parameter (typically 1). 
/// ReLU is computationally efficient and helps mitigate the vanishing gradient problem, allowing for faster training of deep networks.</para>
/// <para><b>Remarks:</b> With beta = 1 (default), this is the standard ReLU. The beta parameter allows for scaling the positive region, 
/// though this is rarely modified in practice. ReLU can suffer from the "dying ReLU" problem where neurons can become permanently inactive 
/// if they always receive negative inputs. Despite this, ReLU and its variants remain the go-to activation for most hidden layers in modern architectures. 
/// ReLU is non-differentiable at x = 0, but in practice, the gradient is defined as 0 or 1 at this point.</para>
/// </remarks>
/// <param name="beta">The scaling factor for positive inputs. Default is 1.</param>
public class ReLU2D(float beta = 1f) : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcOutput(bool inference)
        => ReLUOutput(Input, beta);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => ReLUInputGradient(outputGradient, Input, beta);

    public override string ToString()
        => $"ReLU2D (beta={beta})";
}
