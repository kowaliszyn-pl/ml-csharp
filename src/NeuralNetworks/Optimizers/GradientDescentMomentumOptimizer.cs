// Neural Networks in C♯
// File name: GradientDescentMomentumOptimizer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.LearningRates;

namespace NeuralNetworks.Optimizers;

/// <summary>
/// Implements the classic "Stochastic" Gradient Descent (SGD) optimizer with momentum for neural network training.
/// Updates parameters by applying momentum to the previous update and subtracting the scaled gradient using a specified learning rate.
/// </summary>
/// <remarks>
/// The momentum term helps accelerate gradients in the relevant direction and dampens oscillations.
/// </remarks>
public class GradientDescentMomentumOptimizer(LearningRate learningRate, float momentum) : Optimizer(learningRate)
{
    private readonly Dictionary<object, float[]> _velocities = [];

    /// <summary>
    /// Updates the parameter values in-place using the provided gradients and the optimizer's momentum and learning
    /// rate settings.
    /// </summary>
    /// <remarks>This method applies a momentum-based update to each parameter. The optimizer maintains a
    /// separate velocity vector for each parameter key, which is updated and used to compute the parameter adjustments.
    /// The method modifies the contents of <paramref name="paramsToUpdate"/> directly.</remarks>
    /// <param name="paramsKey">The key that uniquely identifies the parameter set to update. Used to maintain optimizer state for each
    /// parameter group.</param>
    /// <param name="paramsToUpdate">A span containing the parameter values to be updated. The values are modified in-place based on the computed
    /// updates.</param>
    /// <param name="paramGradients">A read-only span containing the gradients corresponding to each parameter in <paramref name="paramsToUpdate"/>.
    /// Must have the same length as <paramref name="paramsToUpdate"/>.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected override void Update(object paramsKey, Span<float> paramsToUpdate, ReadOnlySpan<float> paramGradients)
    {
        int length = paramsToUpdate.Length;
        Debug.Assert(length == paramGradients.Length, "Parameter and gradient spans must have the same length.");

        float learningRate = LearningRate.GetLearningRate();

        // Get or create velocities for the given parameter key. This will be used to store the momentum term for each parameter.

        Span<float> velocities;
        if (_velocities.TryGetValue(paramsKey, out float[]? velocitiesAsArray))
        {
            velocities = velocitiesAsArray;
        }
        else
        {
            velocitiesAsArray = new float[length];
            _velocities.Add(paramsKey, velocitiesAsArray);
            velocities = velocitiesAsArray;
        }

        Debug.Assert(length == velocities.Length, "Parameter and velocity spans must have the same length.");

        
        for (int i = 0; i < length; i++)
        {
            float v = velocities[i] * momentum + learningRate * paramGradients[i];
            velocities[i] = v;
            paramsToUpdate[i] -= v;
        }
    }

    public override string ToString()
        => $"GradientDescentMomentum (learningRate={LearningRate}, momentum={momentum})";
}