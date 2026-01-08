// Neural Networks in C♯
// File name: Linear.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// "Identity" activation function
/// </summary>
public class Linear : ActivationFunction2D
{
    protected override float[,] CalcOutput(bool inference) 
        => Input;

    protected override float[,] CalcInputGradient(float[,] outputGradient) 
        => outputGradient;

    public override string ToString() 
        => "Linear";
}
