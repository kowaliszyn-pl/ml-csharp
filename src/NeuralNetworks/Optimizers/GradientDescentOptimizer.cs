// Neural Networks in C♯
// File name: GradientDescentOptimizer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core.Extensions;
using NeuralNetworks.Layers;
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
    /// Updates a 1D parameter array using the gradient and the current learning rate.
    /// </summary>
    /// <param name="layer">The layer whose parameters are being updated (may be null).</param>
    /// <param name="param">The parameter array to update.</param>
    /// <param name="paramGradient">The gradient array corresponding to <paramref name="param"/>.</param>
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

    /// <summary>
    /// Updates a 2D parameter array using the gradient and the current learning rate.
    /// </summary>
    /// <param name="layer">The layer whose parameters are being updated (may be null).</param>
    /// <param name="param">The 2D parameter array to update.</param>
    /// <param name="paramGradient">The 2D gradient array corresponding to <paramref name="param"/>.</param>
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

    /// <summary>
    /// Updates a 4D parameter array using the gradient and the current learning rate.
    /// </summary>
    /// <param name="layer">The layer whose parameters are being updated (may be null).</param>
    /// <param name="param">The 4D parameter array to update.</param>
    /// <param name="paramGradient">The 4D gradient array corresponding to <paramref name="param"/>.</param>
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

    /// <summary>
    /// Returns a string representation of the optimizer, including the learning rate.
    /// </summary>
    /// <returns>A string describing the optimizer and its learning rate.</returns>
    public override string ToString()
        => $"GradientDescent (learningRate={LearningRate})";
}
