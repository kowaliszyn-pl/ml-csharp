// Neural Networks in C♯
// File name: GradientDescentMomentumOptimizer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;

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
        int length = param.Length;
        for (int i = 0; i < length; i++)
        {
            velocities[i] = velocities[i] * momentum + learningRate * paramGradient[i];
            param[i] -= velocities[i];
        }
    }

    public override void Update(Layer? layer, float[,] param, float[,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

        float[,] velocities = GetOrCreateVelocities(param);

        int dim1 = param.GetLength(0);
        int dim2 = param.GetLength(1);
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                velocities[i, j] = velocities[i, j] * momentum + learningRate * paramGradient[i, j];
                param[i, j] -= velocities[i, j];
            }
        }
    }

    public override void Update(Layer? layer, float[,,,] param, float[,,,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

        float[,,,] velocities = GetOrCreateVelocities(param);

        int dim1 = param.GetLength(0);
        int dim2 = param.GetLength(1);
        int dim3 = param.GetLength(2);
        int dim4 = param.GetLength(3);
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        velocities[i, j, k, l] = velocities[i, j, k, l] * momentum + learningRate * paramGradient[i, j, k, l];
                        param[i, j, k, l] -= velocities[i, j, k, l];
                    }
                }
            }
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
            velocities = new float[param.GetLength(0), param.GetLength(1)];
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
            velocities = new float[param.GetLength(0), param.GetLength(1), param.GetLength(2), param.GetLength(3)];
            _velocities4D.Add(param, velocities);
            return velocities;
        }
    }

    public override string ToString()
        => $"GradientDescentMomentum (learningRate={LearningRate}, momentum={momentum})";
}