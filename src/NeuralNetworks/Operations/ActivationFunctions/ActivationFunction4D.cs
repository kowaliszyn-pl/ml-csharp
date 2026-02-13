// Neural Networks in C♯
// File name: ActivationFunction4D.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Defines an abstract 4D activation function operation applied element-wise to convolutional tensors.
/// </summary>
/// <remarks>
/// Activation functions introduce non-linearity into neural network computations. This base class specifies
/// the contract for operations that transform 4D tensors, typically shaped as <code>(batch, channels, height, width)</code>
/// in convolutional pipelines. Concrete implementations should provide the forward activation and, where applicable,
/// support gradient computation for backpropagation by deriving from <see cref="Operation4D"/>.
/// </remarks>
public abstract class ActivationFunction4D : Operation<float[,,,], float[,,,]>
{
}
