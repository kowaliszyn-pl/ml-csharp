// Neural Networks in C♯
// File name: Dropout3D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

using NeuralNetworks.Core;

namespace NeuralNetworks.Operations.Dropouts;

public class Dropout3D(float keepProb = 0.8f, SeededRandom? random = null) : BaseDropout<float[,,]>
{
    protected override float[,,] CalcInputGradient(float[,,] outputGradient) => throw new NotImplementedException();
    protected override float[,,] CalcOutput(bool inference) => throw new NotImplementedException();
}
