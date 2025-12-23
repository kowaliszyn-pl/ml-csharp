// Neural Networks in C♯
// File name: ActivationFunction2D.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Represents a 2D activation function operation within the neural network pipeline.
/// </summary>
/// <remarks>
/// This abstract base class defines the contract for activation functions that operate on two-dimensional data,
/// such as feature maps in convolutional neural networks. Implementations should provide the forward activation
/// transformation and, where applicable, support gradient computation for backpropagation.
/// </remarks>
public abstract class ActivationFunction2D : Operation2D
{
}
