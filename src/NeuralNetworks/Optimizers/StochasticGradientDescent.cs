// Neural Networks in C♯
// File name: StochasticGradientDescent.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.LearningRates;

namespace NeuralNetworks.Optimizers;

public class StochasticGradientDescent(LearningRate learningRate) : Optimizer(learningRate)
{
    public override void Update(Layer? layer, float[] param, float[] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();
        int length = param.Length;
        for (int i = 0; i < length; i++)
        {
            param[i] -= learningRate * paramGradient[i];
        }
    }

    public override void Update(Layer? layer, float[,] param, float[,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

        int dim1 = param.GetLength(0);
        int dim2 = param.GetLength(1);
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                param[i, j] -= learningRate * paramGradient[i, j];
            }
        }
    }

    public override void Update(Layer? layer, float[,,,] param, float[,,,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

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
                        param[i, j, k, l] -= learningRate * paramGradient[i, j, k, l];
                    }
                }
            }
        }
    }

    public override string ToString()
        => $"StochasticGradientDescent (learningRate={LearningRate})";

}
