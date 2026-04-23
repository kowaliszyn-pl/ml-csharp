// Neural Networks in C♯
// File name: MeanSquaredErrorLoss4D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Losses;

public class MeanSquaredErrorLoss4D : Loss<float[,,,]>
{
    protected override float CalculateLoss() => throw new NotImplementedException();
    protected override float[,,,] CalculateLossGradient() => throw new NotImplementedException();
    override public string ToString() => "MeanSquaredError4D";
}
