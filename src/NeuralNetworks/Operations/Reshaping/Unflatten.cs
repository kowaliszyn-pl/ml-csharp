// Neural Networks in C♯
// File name: Unflatten.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Reshaping;

public class Unflatten(int channels, int height, int width) : Operation<float[,], float[,,,]>
{
    protected override float[,,,] CalcOutput(bool inference) 
        => Unflatten(Input, channels, height, width);

    protected override float[,] CalcInputGradient(float[,,,] outputGradient)
        => Flatten(outputGradient);

    public override string ToString()
        => "Unflatten";
}
