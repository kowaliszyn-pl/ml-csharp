// Neural Networks in C♯
// File name: TanhScaled2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Input-Scaled Hyperbolic Tangent activation function for 2D tensors.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = tanh(x / scale)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · (1 - f(x)²) / scale = ∂L/∂y · (1 - tanh²(x / scale)) / scale</para>
/// <para><b>Output Range:</b> (-1, 1)</para>
/// <para><b>Description:</b> A variant of the standard <see cref="Tanh2D"/> function that scales the input before applying the hyperbolic tangent. 
/// The scale parameter controls the "steepness" of the activation - larger scale values result in a smoother, more gradual transition, 
/// while smaller values create a sharper transition similar to a step function. This can help control gradient flow during training.</para>
/// <para><b>Remarks:</b> The reciprocal of the scale is precomputed for efficiency, as multiplication is faster than division. 
/// Input scaling can help with gradient stability and allows for fine-tuning of the activation's sensitivity to input variations. 
/// When scale = 1, this reduces to the standard tanh function. Larger scale values (e.g., 2-10) can help prevent gradient saturation 
/// by keeping inputs in the more linear region of tanh.</para>
/// </remarks>
/// <param name="scale">The scaling factor applied to the input before the tanh operation. Must be non-zero.</param>
public class TanhInputScaled2D(float scale) : ActivationFunction<float[,], float[,]>
{

    protected override float[,] CalcOutput(bool inference)
        => TanhInputScaledOutput(Input, scale);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => TanhInputScaledInputGradient(outputGradient, Output, scale);

    public override string ToString()
        => $"TanhScaled2D (scale={scale})";

}
