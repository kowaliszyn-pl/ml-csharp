// Neural Networks in C♯
// File name: ActivationFunction4D.cs
// www.kowaliszyn.pl, 2025

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Represents an abstract 4-dimensional activation function operation.
/// </summary>
/// <remarks>
/// Activation functions introduce non-linearity into neural network computations.
/// This base class defines the contract for operations that transform 4D tensors,
/// typically in the shape of <c>(batch, channels, height, width)</c>.
/// Concrete implementations should provide the forward and (optionally) backward
/// passes by deriving from <c>Operation4D</c>.
/// </remarks>
/// <seealso cref="Operation4D"/>
public abstract class ActivationFunction4D : Operation4D
{
}
