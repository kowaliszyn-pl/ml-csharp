// Neural Networks in C♯
// File name: BaseDropout2D.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Operations.Dropouts;

public abstract class BaseDropout2D : Operation2D
{
    protected float[,]? Mask { get; set; }
}
