// Neural Networks in C♯
// File name: GradientDescentMomentumOptimizer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.LearningRates;

namespace NeuralNetworks.Optimizers;

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

        //if (_velocities1D.ContainsKey(param))
        //{
        //    return _velocities1D[param];
        //}
        //else
        //{
        //    float[] velocities = new float[param.Length];
        //    _velocities1D.Add(param, velocities);
        //    return velocities;
        //}
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
        => $"StochasticGradientDescentMomentum (learningRate={LearningRate}, momentum={momentum})";
}

/*
 
using System.Diagnostics;
using System.Runtime.CompilerServices;

using MachineLearning.GenericModel.LearningRates;
using MachineLearning.Typed.GenericModel.Layers;

namespace MachineLearning.Typed.GenericModel.Optimizers;

public class GradientDescentMomentumOptimizer(LearningRate learningRate, float momentum) : Optimizer(learningRate)
{
    private Dictionary<Array, Array> _velocities = new();

    public override void Update(Layer layer, float[] param, float[] paramGradient)
    {
        UpdateParameters(param, paramGradient, momentum);
    }

    public override void Update(Layer layer, float[,] param, float[,] paramGradient)
    {
        UpdateParameters(param, paramGradient, momentum);
    }

    public override void Update(Layer layer, float[,,,] param, float[,,,] paramGradient)
    {
        UpdateParameters(param, paramGradient, momentum);
    }

    private void UpdateParameters<T>(T param, T paramGradient, float momentum) where T : Array
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();
        T velocities = GetOrCreateVelocities(param);

        int[] indices = new int[param.Rank];
        UpdateRecursive(param, paramGradient, velocities, learningRate, momentum, indices, 0);
    }

    private void UpdateRecursive<T>(T param, T paramGradient, T velocities, float learningRate, float momentum, int[] indices, int dimension) where T : Array
    {
        int length = param.GetLength(dimension);
        for (int i = 0; i < length; i++)
        {
            indices[dimension] = i;
            if (dimension == param.Rank - 1)
            {
                float velocity = (float)velocities.GetValue(indices);
                float gradient = (float)paramGradient.GetValue(indices);
                velocity = velocity * momentum + learningRate * gradient;
                velocities.SetValue(velocity, indices);
                param.SetValue((float)param.GetValue(indices) - velocity, indices);
            }
            else
            {
                UpdateRecursive(param, paramGradient, velocities, learningRate, momentum, indices, dimension + 1);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private T GetOrCreateVelocities<T>(T param) where T : Array
    {
        if (_velocities.TryGetValue(param, out Array? velocities))
        {
            return (T)velocities;
        }
        else
        {
            T newVelocities = (T)Activator.CreateInstance(typeof(T), param.GetLengths());
            _velocities.Add(param, newVelocities);
            return newVelocities;
        }
    }

    public override string ToString() => $"GradientDescentMomentumOptimizer (learningRate={LearningRate}, momentum={momentum})";
}


 */