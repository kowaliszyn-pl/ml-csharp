// Neural Networks in C♯
// File name: ReLU.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Rectified Linear Unit (ReLU) activation function for 4D tensors.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = max(0, beta · x)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · beta if x > 0, else 0</para>
/// <para><b>Output Range:</b> [0, ∞) when beta > 0</para>
/// <para><b>Description:</b> ReLU variant for 4D tensors, primarily used in convolutional neural networks (CNNs) where tensors 
/// typically represent [batch, channels, height, width]. Provides the same benefits as standard ReLU - computational efficiency, 
/// mitigation of vanishing gradients, and sparse activation - while supporting batch processing of multi-channel image data.</para>
/// <para><b>Remarks:</b> Most commonly used in CNNs for computer vision tasks. The beta parameter (default 1) allows scaling of positive inputs, 
/// though the standard value is almost universally used. Like other ReLU variants, it can suffer from the dying ReLU problem, 
/// but alternatives like Leaky ReLU or ELU can address this if needed. The 4D implementation enables efficient batch processing typical in modern deep learning frameworks.</para>
/// </remarks>
/// <param name="beta">The scaling factor for positive inputs. Default is 1.</param>
public class ReLU4D(float beta = 1f) : ActivationFunction<float[,,,], float[,,,]>
{
    protected override float[,,,] CalcOutput(bool inference)
        => ReLUOutput(Input, beta);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
        => ReLUInputGradient(outputGradient, Input, beta);

    public override string ToString()
        => $"ReLU4D (beta={beta})";
}
