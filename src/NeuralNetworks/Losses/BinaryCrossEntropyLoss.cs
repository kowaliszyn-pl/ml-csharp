// Neural Networks in C♯
// File name: BinaryCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Losses;

public class BinaryCrossEntropyLoss : Loss<float[,]>
{
    protected override float CalculateLoss() => throw new NotImplementedException();
    protected override float[,] CalculateLossGradient() => throw new NotImplementedException();
}
