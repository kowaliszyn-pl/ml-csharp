// Neural Networks in C♯
// File name: BaseDropout2D.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Operations.Interfaces;

namespace NeuralNetworks.Operations;

public abstract class BaseDropout2D : Operation2D, IParameterCountProvider
{
    protected float[,]? Mask { get; set; }

    public int GetParamCount()
        => Mask?.Length ?? 0;
}
