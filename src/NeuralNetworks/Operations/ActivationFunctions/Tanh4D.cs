// Neural Networks in C♯
// File name: Tanh4D.cs
// www.kowaliszyn.pl, 2025


// Neural Networks in C♯
// File name: Tanh4D.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class Tanh4D : ActivationFunction4D
{
    protected override float[,,,] CalcOutput(bool inference) 
        => Input.Tanh();

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Tanh function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Tanh function tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) is 1 - tanh(x)^2.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * (1 - tanh(x)^2).
        // The elementwise multiplication of the output gradient and the derivative of the Tanh function is returned as the input gradient.
        // tanh(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient
        
        //float[,,,] tanhBackward = Output.AsOnes().Subtract(Output.MultiplyElementwise(Output));
        //return outputGradient.MultiplyElementwise(tanhBackward);

        return outputGradient.MultiplyByTanhDerivative(Output);
    }

    public override string ToString() => "Tanh4D";
}
