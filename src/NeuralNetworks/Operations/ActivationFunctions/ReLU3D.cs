// Neural Networks in C♯
// File name: ReLU1D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class ReLU3D : ActivationFunction<float[,,], float[,,]>
{
    protected override float[,,] CalcInputGradient(float[,,] outputGradient) => throw new NotImplementedException();
    protected override float[,,] CalcOutput(bool inference) => throw new NotImplementedException();
}
