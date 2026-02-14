// Neural Networks in C♯
// File name: GradientDescentOptimizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

using NeuralNetworks.LearningRates;

namespace NeuralNetworks.Optimizers;

/// <summary>
/// Implements the classic "Stochastic" Gradient Descent (SGD) optimizer for neural network training.
/// Updates parameters by subtracting the scaled gradient using a specified learning rate.
/// </summary>
/// <remarks>
/// This optimizer supports parameter updates for 1D, 2D, and 4D float arrays.
/// <para/>
/// There is no actual "stochastic" aspect implemented here, so the name of this class reflects that ("GradientDescentOptimizer" instead of "StochasticGradientDescentOptimizer").
/// </remarks>
public class GradientDescentOptimizer(LearningRate learningRate) : Optimizer(learningRate)
{
    /// <summary>
    /// Updates the parameter values in-place by applying the provided gradients using the current learning rate.
    /// </summary>
    /// <remarks>
    /// This method performs a simple gradient descent update. The learning rate is obtained from the
    /// associated learning rate schedule. The update is applied element-wise to each parameter value.
    /// </remarks>
    /// <param name="paramsKey">Not used in this class.</param>
    /// <param name="paramsToUpdate">A span representing the parameter values to be updated. The values are modified in-place.</param>
    /// <param name="paramGradients">A read-only span containing the gradients to apply to the parameter values. Each element corresponds to the
    /// respective parameter in <paramref name="paramsToUpdate"/>.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected override void Update(object paramsKey, Span<float> paramsToUpdate, ReadOnlySpan<float> paramGradients)
    {
        Debug.Assert(paramsToUpdate.Length == paramGradients.Length, "Parameter and gradient spans must have the same length.");

        float learningRate = LearningRate.GetLearningRate();
        for (int i = 0; i < paramsToUpdate.Length; i++)
        {
            paramsToUpdate[i] -= learningRate * paramGradients[i];
        }
    }

    /// <summary>
    /// Returns a string representation of the optimizer, including the learning rate.
    /// </summary>
    /// <returns>A string describing the optimizer and its learning rate.</returns>
    public override string ToString()
        => $"GradientDescent (learningRate={LearningRate})";

}
