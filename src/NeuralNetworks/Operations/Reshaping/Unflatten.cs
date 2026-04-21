// Neural Networks in C♯
// File name: Unflatten.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Operations.Reshaping;

public class Unflatten : Operation<float[,], float[,,,]>
{
    protected override float[,] CalcInputGradient(float[,,,] outputGradient) => throw new NotImplementedException();

    protected override float[,,,] CalcOutput(bool inference) => throw new NotImplementedException();

    public override string ToString()
        => "Unflatten";
}
