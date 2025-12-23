// Neural Networks in C♯
// File name: ActivationFunction2D.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Defines an abstract 2D activation function operation applied element-wise to two-dimensional data.
/// </summary>
/// <remarks>
/// Activation functions introduce non-linearity into neural network computations. This base class specifies
/// the contract for operations that transform 2D tensors (e.g., matrices or 2D feature maps), and are typically
/// used in layers processing spatial data without an explicit channel dimension. Concrete implementations should
/// provide the forward activation and, where applicable, support gradient computation for backpropagation by
/// deriving from <see cref="Operation2D"/>.
/// </remarks>
public abstract class ActivationFunction2D : Operation2D
{
}
