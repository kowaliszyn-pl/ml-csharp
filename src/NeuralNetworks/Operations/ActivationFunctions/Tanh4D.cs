// Neural Networks in C♯
// File name: Tanh4D.cs
// www.kowaliszyn.pl, 2025

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class Tanh4D : ActivationFunction4D
{
    protected override float[,,,] CalcOutput(bool inference)
        => Tanh(Input);

    /// <summary>
    /// Calculates the gradient of the loss with respect to the input of the Tanh activation function.
    /// </summary>
    /// <remarks>
    /// This method is used during backpropagation in neural network training to propagate
    /// gradients through a Tanh activation layer. The returned array has the same shape as <paramref
    /// name="outputGradient"/>.
    /// <para>
    /// This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx). The derivative of the Tanh function is <c>1 - tanh(x)^2</c>. Therefore, the input gradient is computed as: <c>dL/dx = dL/dy * (1 - tanh(x)^2)</c>. The elementwise multiplication of the output gradient and the derivative of the Tanh function is returned as the input gradient.
    /// </para>
    /// <list type="bullet">
    /// <item>
    /// tanh(x) => Output
    /// </item>
    /// <item>
    /// dL/dy => outputGradient
    /// </item>
    /// <item>
    /// dl/dx => inputGradient
    /// </item>
    /// </list>
    /// </remarks>
    /// <param name="outputGradient">A four-dimensional array representing the gradient of the loss with respect to the output of the Tanh function. The shape must match the output tensor of the layer.</param>
    /// <returns>A four-dimensional array containing the gradient of the loss with respect to the input of the Tanh function.
    /// Each element is computed by multiplying the corresponding element in <paramref name="outputGradient"/> by the
    /// derivative of the Tanh function at that position.
    /// </returns>
    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient) 
        => MultiplyByTanhDerivative(outputGradient, Output);

    public override string ToString() 
        => "Tanh4D";
}
