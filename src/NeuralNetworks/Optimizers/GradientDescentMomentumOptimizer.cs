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
/// This optimizer supports parameter updates for 1D, 2D, and 4D float arrays.
/// The momentum term helps accelerate gradients in the relevant direction and dampens oscillations.
/// </remarks>
public class GradientDescentMomentumOptimizer(LearningRate learningRate, float momentum) : Optimizer(learningRate)
{
    private readonly Dictionary<float[], float[]> _velocities1D = [];
    private readonly Dictionary<float[,], float[,]> _velocities2D = [];
    private readonly Dictionary<float[,,,], float[,,,]> _velocities4D = [];

    public override void Update(Layer? layer, float[] param, float[] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

        float[] velocities = GetOrCreateVelocities(param);

        ApplyMomentumUpdate(
            param.AsSpan(),
            paramGradient.AsSpan(),
            velocities.AsSpan(),
            learningRate
        );
    }

    public override void Update(Layer? layer, float[,] param, float[,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

        float[,] velocities = GetOrCreateVelocities(param);

        ApplyMomentumUpdate(
             MemoryMarshal.CreateSpan(ref param[0, 0], param.Length),
             MemoryMarshal.CreateReadOnlySpan(ref paramGradient[0, 0], paramGradient.Length),
             MemoryMarshal.CreateSpan(ref velocities[0, 0], velocities.Length),
             learningRate
        );
    }

    public override void Update(Layer? layer, float[,,,] param, float[,,,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

        float[,,,] velocities = GetOrCreateVelocities(param);

        ApplyMomentumUpdate(
            MemoryMarshal.CreateSpan(ref param[0, 0, 0, 0], param.Length),
            MemoryMarshal.CreateReadOnlySpan(ref paramGradient[0, 0, 0, 0], paramGradient.Length),
            MemoryMarshal.CreateSpan(ref velocities[0, 0, 0, 0], velocities.Length),
            learningRate
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyMomentumUpdate(Span<float> parameters, ReadOnlySpan<float> gradients, Span<float> velocities, float learningRate)
    {
        Debug.Assert(parameters.Length == gradients.Length);
        Debug.Assert(parameters.Length == velocities.Length);

        int length = parameters.Length;
        for (int i = 0; i < length; i++)
        {
            float v = velocities[i] * momentum + learningRate * gradients[i];
            velocities[i] = v;
            parameters[i] -= v;
        }
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float[] GetOrCreateVelocities(float[] param)
    {
        if (_velocities1D.TryGetValue(param, out float[]? velocities))
        {
            return velocities;
        }
        else
        {
            velocities = new float[param.Length];
            _velocities1D.Add(param, velocities);
            return velocities;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float[,] GetOrCreateVelocities(float[,] param)
    {
        if (_velocities2D.TryGetValue(param, out float[,]? velocities))
        {
            return velocities;
        }
        else
        {
            velocities = param.AsZeros();
            _velocities2D.Add(param, velocities);
            return velocities;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float[,,,] GetOrCreateVelocities(float[,,,] param)
    {
        if (_velocities4D.TryGetValue(param, out float[,,,]? velocities))
        {
            return velocities;
        }
        else
        {
            velocities = param.AsZeros();
            _velocities4D.Add(param, velocities);
            return velocities;
        }
    }

    public override string ToString()
        => $"GradientDescentMomentum (learningRate={LearningRate}, momentum={momentum})";
}