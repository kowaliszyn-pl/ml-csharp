// Neural Networks in C♯
// File name: ActivationUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Utils;

public static class ActivationUtils
{
    /// <summary>
    /// Applies the Gaussian Error Linear Unit (GELU) activation function to each element of the specified
    /// two-dimensional array.
    /// </summary>
    /// <remarks>
    /// The GELU activation function is commonly used in neural networks and is defined as GELU(x) = 0.5 * x * (1 +
    /// tanh(√(2/π) * (x + 0.044715 * x^3))). This method does not modify the input array; it returns a new array
    /// containing the transformed values.
    /// </remarks>
    /// <param name="source">
    /// A two-dimensional array of single-precision floating-point values to which the GELU function will be applied.
    /// </param>
    /// <returns>
    /// A two-dimensional array of the same dimensions as the input, where each element is the result of applying the
    /// GELU function to the corresponding element in the input array.
    /// </returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Gelu(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        const float sqrt2OverPi = 0.7978845608028654f; // sqrt(2 / pi)
        const float coeff = 0.044715f;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float x = source[i, j];
                float xCubed = x * x * x;
                float tanhArg = sqrt2OverPi * (x + coeff * xCubed);
                float tanhValue = MathF.Tanh(tanhArg);
                res[i, j] = 0.5f * x * (1 + tanhValue);
            }
        }
        return res;
    }

    /// <summary>
    /// Computes the log-softmax of each row in a two-dimensional array of single-precision floating-point values.
    /// </summary>
    /// <remarks>
    /// The log-softmax function is useful for numerical stability when working with probabilities in machine learning models.
    /// It is computed as the logarithm of the softmax function, which ensures that the output values are normalized and
    /// sum to 1 in the probability space.
    /// </remarks>
    /// <param name="source">The two-dimensional array to transform (log-softmax applied per row).</param>
    /// <returns>A new source with log-softmax-applied values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] LogSoftmax(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++) // dim1 = batch size (observations)
        {
            float max = source[i, 0];
            for (int j = 1; j < columns; j++)
            {
                max = MathF.Max(max, source[i, j]);
            }

            float sum = 0;
            for (int j = 0; j < columns; j++)
            {
                sum += MathF.Exp(source[i, j] - max);
            }

            float logSumExp = max + MathF.Log(sum);

            for (int j = 0; j < columns; j++)
            {
                res[i, j] = source[i, j] - logSumExp;
            }
        }

        Debug.Assert(res.Cast<float>().All(x => !float.IsNaN(x)), "There should be no NaN values");

        return res;
    }

    /// <summary>
    /// Applies the sigmoid function to each element of the source.
    /// </summary>
    /// <returns>
    /// A new source with each element transformed by the sigmoid function with the same dimensions as the original
    /// source.
    /// </returns>
    /// <param name="source">The two-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Sigmoid(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = 1 / (1 + MathF.Exp(-source[i, j]));
            }
        }

        return res;
    }

    /// <summary>
    /// Calculates the derivative of the sigmoid function for each element of the source.
    /// </summary>
    /// <remarks>
    /// The derivative of the sigmoid function is calculated as: sigmoid(x) * (1 - sigmoid(x)).
    /// </remarks>
    /// <returns>
    /// A new source with each element transformed by the derivative of the sigmoid function with the same dimensions as
    /// the original source.
    /// </returns>
    /// <param name="source">The two-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] SigmoidDerivative(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float sigmoid = 1 / (1 + MathF.Exp(-source[i, j]));
                res[i, j] = sigmoid * (1 - sigmoid);
            }
        }

        return res;
    }

    /// <summary>
    /// Applies the softmax function to the source.
    /// </summary>
    /// <returns>A new source with softmax-applied values.</returns>
    /// <remarks>Softmax formula: <c>exp(x) / sum(exp(x))</c>.</remarks>
    /// <param name="source">The two-dimensional array to transform (softmax applied per row).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] SoftmaxBasic(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] res = new float[rows, columns];

        float[] rowExponentials = new float[columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                rowExponentials[j] = MathF.Exp(source[i, j]);
            }

            float sum = 0;
            for (int j = 0; j < columns; j++)
            {
                sum += rowExponentials[j];
            }

            for (int j = 0; j < columns; j++)
            {
                res[i, j] = rowExponentials[j] / sum;
            }
        }

        Debug.Assert(res.Cast<float>().All(x => !float.IsNaN(x)), "There should be no NaN values");

        return res;
    }

    /// <summary>
    /// Computes the numerically stable softmax of each row in a two-dimensional array of single-precision
    /// floating-point values.
    /// </summary>
    /// <remarks>
    /// This method applies the softmax function to each row independently, using a numerically stable algorithm to
    /// prevent overflow or underflow. The input array is not modified.
    /// </remarks>
    /// <param name="source">
    /// A two-dimensional array of type <see cref="float"/> representing the input data. Each row is treated as a
    /// separate vector for which the softmax will be computed.
    /// </param>
    /// <returns>
    /// A two-dimensional array of type <see cref="float"/> containing the softmax probabilities for each row of the
    /// input array. Each row sums to 1.0.
    /// </returns>
    public static float[,] SoftmaxStable(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            // 1. Find the max for numerical stability
            float max = source[i, 0];
            for (int j = 1; j < columns; j++)
            {
                if (source[i, j] > max)
                    max = source[i, j];
            }

            // 2. Exponentiate shifted values and compute sum
            float sumExp = 0f;
            for (int j = 0; j < columns; j++)
            {
                float exp = MathF.Exp(source[i, j] - max);
                res[i, j] = exp;
                sumExp += exp;
            }

            // 3. Normalize
            float invSumExp = 1.0f / sumExp;
            for (int j = 0; j < columns; j++)
            {
                res[i, j] *= invSumExp; // multiplication is faster than division
            }
        }

        return res;
    }

    public static float[] SoftmaxStableWithTemperature(this float[] source, float temperature)
    {
        Debug.Assert(temperature > 0, "Temperature must be greater than 0.");

        int length = source.GetLength(0);
        float[] res = new float[length];

        float max = float.NegativeInfinity;
        for (int i = 0; i < length; i++)
        {
            // 1. Find the max for numerical stability
            if (source[i] > max)
                max = source[i];
        }

        // 2. Exponentiate shifted values and compute sum
        float sum = 0f;
        for (int i = 0; i < length; i++)
        {
            float exp = MathF.Exp((source[i] - max) / temperature);
            res[i] = exp;
            sum += exp;
        }

        // 3. Normalize
        for (int i = 0; i < length; i++)
        {
            res[i] /= sum;
        }

        return res;
    }

    /// <summary>
    /// Applies the Softplus activation function to each element of the specified two-dimensional array.
    /// </summary>
    /// <remarks>
    /// The Softplus function is defined as <c> log(1 + exp(x))</c> and is commonly used as a smooth approximation to
    /// the ReLU activation in machine learning applications. The returned array is a new instance; the input array is
    /// not modified.
    /// </remarks>
    /// <param name="source">
    /// A two-dimensional array of single-precision floating-point values to which the Softplus function will be
    /// applied.
    /// </param>
    /// <returns>
    /// A two-dimensional array of the same dimensions as <paramref name="source"/>, where each element is the result of
    /// applying the Softplus function to the corresponding element in <paramref name="source"/>.
    /// </returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Softplus(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = MathF.Log(1 + MathF.Exp(source[i, j]));
            }
        }
        return res;
    }

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the source.
    /// </summary>
    /// <returns>A new source with the hyperbolic tangent applied element-wise.</returns>
    /// <param name="source">The two-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Tanh(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = MathF.Tanh(source[i, j]);
            }
        }

        return res;
    }

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the source.
    /// </summary>
    /// <returns>A new source with the hyperbolic tangent applied element-wise.</returns>
    /// <param name="source">The four-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Tanh(this float[,,,] source)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        res[i, j, k, l] = MathF.Tanh(source[i, j, k, l]);
                    }
                }
            }
        }

        return res;
    }
}
