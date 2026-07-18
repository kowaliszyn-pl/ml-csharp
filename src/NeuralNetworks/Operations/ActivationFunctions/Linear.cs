// Neural Networks in C♯
// File name: Linear.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Linear (Identity) activation function.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = x</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · 1 = ∂L/∂y</para>
/// <para><b>Output Range:</b> (-∞, ∞)</para>
/// <para><b>Description:</b> The linear activation function simply returns its input without any transformation. 
/// It is primarily used in regression tasks where the output layer needs to predict continuous values without bounds. 
/// Since it provides no non-linearity, using linear activations in all layers would collapse a deep network into a single-layer linear model.</para>
/// <para><b>Remarks:</b> The linear activation is also known as the "identity" function. 
/// It has a constant gradient of 1, which means it does not suffer from vanishing or exploding gradient problems by itself. 
/// However, stacking multiple linear layers without non-linear activations in between provides no benefit over a single linear layer, 
/// as the composition of linear functions is still linear. Linear activation is typically used only in the output layer for regression tasks 
/// or in specific architectural components where no transformation is desired (e.g., residual connections in ResNets use identity mappings).</para>
/// </remarks>
public class Linear : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcOutput(bool inference) 
        => Input;

    protected override float[,] CalcInputGradient(float[,] outputGradient) 
        => outputGradient;

    public override string ToString() 
        => "Linear";
}
