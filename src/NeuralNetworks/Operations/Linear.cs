// Machine Learning Utils
// File name: Linear.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.Operations;

/// <summary>
/// "Identity" activation function
/// </summary>
public class Linear : Operation2D
{
    protected override float[,] CalcOutput(bool inference) => Input;

    protected override float[,] CalcInputGradient(float[,] outputGradient) => outputGradient;

    public override string ToString() => "Linear";
}
