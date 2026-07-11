// Neural Networks in C♯
// File name: Upsample2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Reshaping;

public class Upsample2D(int scaleHeight, int scaleWidth) : Operation<float[,,,], float[,,,]>
{
    protected override float[,,,] CalcOutput(bool inference)
       => Upsample2DOutput(Input, scaleHeight, scaleWidth);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
        => Upsample2DInputGradient(Input, outputGradient);

}
