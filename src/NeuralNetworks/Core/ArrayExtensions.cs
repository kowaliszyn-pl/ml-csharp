// Neural Networks in C♯
// File name: ArrayExtensions.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

/// <summary>
/// Provides extension methods for performing mathematical and utility operations on multidimensional arrays of type
/// <see cref="float"/>. These methods enable elementwise arithmetic, statistical calculations, source manipulation, and
/// other common operations for arrays and matrices.
/// </summary>
/// <remarks>
/// The <see cref="ArrayExtensions"/> class includes methods for both in-place and out-of-place operations, supporting
/// 1D, 2D, and 4D float arrays. It offers functionality such as elementwise addition, multiplication, subtraction,
/// standardization, transposition, and application of activation functions (e.g., sigmoid, softmax, tanh). Methods are
/// designed to simplify array manipulation in numerical and machine learning scenarios. Thread safety is not
/// guaranteed; callers should synchronize access if arrays are shared across threads.
/// </remarks>
public static class ArrayExtensions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Abs(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = MathF.Abs(source[row, col]);
            }
        }
        return res;
    }

    /// <summary>
    /// Computes the index of the maximum value for each row.
    /// </summary>
    /// <param name="source">The array to evaluate.</param>
    /// <returns>The index of the maximum value in the array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Argmax(this float[] source)
    {
        int items = source.Length;

        float max = float.MinValue;
        int maxIndex = 0;
        for (int i = 0; i < items; i++)
        {
            float value = source[i];
            if (value > max)
            {
                max = value;
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /// <summary>
    /// Computes the index of the maximum value for each row.
    /// </summary>
    /// <param name="source">The two-dimensional array to evaluate.</param>
    /// <returns>
    /// An array of length equal to the number of rows, where each element is the column index of the maximum value in
    /// that row.
    /// </returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int[] Argmax(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        int[] array = new int[rows];

        for (int row = 0; row < rows; row++)
        {
            float max = float.MinValue;
            int maxIndex = 0;
            for (int col = 0; col < columns; col++)
            {
                float value = source[row, col];
                if (value > max)
                {
                    max = value;
                    maxIndex = col;
                }
            }
            array[row] = maxIndex;
        }

        return array;
    }

    

    /// <summary>
    /// Returns a new two-dimensional array with each element of the source array limited to the specified minimum and
    /// maximum values.
    /// </summary>
    /// <remarks>
    /// If an element in the source array is less than the specified minimum, the result will contain the minimum value
    /// at that position. If an element is greater than the specified maximum, the result will contain the maximum
    /// value. The original array is not modified.
    /// </remarks>
    /// <param name="source">The two-dimensional array of single-precision floating-point numbers to be clipped.</param>
    /// <param name="min">The minimum value to which elements in the array will be limited.</param>
    /// <param name="max">The maximum value to which elements in the array will be limited.</param>
    /// <returns>
    /// A new two-dimensional array where each element is set to the corresponding value from the source array, limited
    /// to the specified minimum and maximum values.
    /// </returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Clip(this float[,] source, float min, float max)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = MathF.Max(min, MathF.Min(max, source[row, col]));
            }
        }
        return res;
    }

    /// <summary>
    /// Clips the values of the source in-place between the specified minimum and maximum values.
    /// </summary>
    /// <param name="source">The two-dimensional array to clip.</param>
    /// <param name="min">The minimum value to clip the source elements to.</param>
    /// <param name="max">The maximum value to clip the source elements to.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ClipInPlace(this float[,] source, float min, float max)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                source[row, col] = MathF.Max(min, MathF.Min(max, source[row, col]));
            }
        }
    }

    /// <summary>
    /// Checks whether two one-dimensional arrays have the same shape (length).
    /// </summary>
    /// <param name="source">The first array.</param>
    /// <param name="matrix">The second array.</param>
    /// <returns>True if both arrays have equal length; otherwise false.</returns>
    public static bool HasSameShape(this float[] source, float[] matrix)
        => source.GetLength(0) == matrix.GetLength(0);

    /// <summary>
    /// Checks whether two two-dimensional arrays have the same shape (dim1 and dim2).
    /// </summary>
    /// <param name="source">The first array.</param>
    /// <param name="matrix">The second array.</param>
    /// <returns>True if both arrays have equal dim1 and dim2; otherwise false.</returns>
    public static bool HasSameShape(this float[,] source, float[,] matrix)
        => source.GetLength(0) == matrix.GetLength(0)
            && source.GetLength(1) == matrix.GetLength(1);

    public static bool HasSameShape(this float[,,] source, float[,,] matrix)
        => source.GetLength(0) == matrix.GetLength(0)
            && source.GetLength(1) == matrix.GetLength(1)
            && source.GetLength(2) == matrix.GetLength(2);

    /// <summary>
    /// Checks whether two four-dimensional arrays have the same shape across all dimensions.
    /// </summary>
    /// <param name="source">The first array.</param>
    /// <param name="matrix">The second array.</param>
    /// <returns>True if all corresponding dimensions are equal; otherwise false.</returns>
    public static bool HasSameShape(this float[,,,] source, float[,,,] matrix)
        => source.GetLength(0) == matrix.GetLength(0)
            && source.GetLength(1) == matrix.GetLength(1)
            && source.GetLength(2) == matrix.GetLength(2)
            && source.GetLength(3) == matrix.GetLength(3);

    /// <summary>
    /// Computes the natural logarithm elementwise and returns a new array.
    /// </summary>
    /// <param name="source">The array whose elements will be transformed.</param>
    /// <returns>A new array with <c>log(x)</c> applied elementwise.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Log(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = MathF.Log(source[i, j]);
            }
        }

        return res;
    }

    /// <summary>
    /// Returns the maximum element value across all elements of a two-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The maximum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(this float[,] source)
    {
        float max = float.MinValue;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                max = Math.Max(max, source[row, col]);
            }
        }
        return max;
    }

    /// <summary>
    /// Returns the maximum element value across all elements of a three-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The maximum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(this float[,,] source)
    {
        float max = float.MinValue;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    max = Math.Max(max, source[i, j, k]);
                }
            }
        }

        return max;
    }

    /// <summary>
    /// Returns the maximum element value across all elements of a four-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The maximum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(this float[,,,] source)
    {
        float max = float.MinValue;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        max = Math.Max(max, source[i, j, k, l]);
                    }
                }
            }
        }

        return max;
    }

    /// <summary>
    /// Calculates the mean of all elements in the source.
    /// </summary>
    /// <param name="source">The array whose mean will be computed.</param>
    /// <returns>The arithmetic mean of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mean(this float[,] source)
        => source.Sum() / source.Length;

    /// <summary>
    /// Calculates the mean of all elements in the source.
    /// </summary>
    /// <param name="source">The array whose mean will be computed.</param>
    /// <returns>The arithmetic mean of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mean(this float[,,] source)
        => source.Sum() / source.Length;

    /// <summary>
    /// Returns the minimum element value across all elements of a two-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The minimum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(this float[,] source)
    {
        float min = float.MaxValue;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                min = Math.Min(min, source[row, col]);
            }
        }
        return min;
    }

    /// <summary>
    /// Returns the minimum element value across all elements of a three-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The minimum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(this float[,,] source)
    {
        float min = float.MaxValue;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    min = Math.Min(min, source[i, j, k]);
                }
            }
        }

        return min;
    }

    /// <summary>
    /// Returns the minimum element value across all elements of a four-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The minimum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(this float[,,,] source)
    {
        float min = float.MaxValue;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        min = Math.Min(min, source[i, j, k, l]);
                    }
                }
            }
        }

        return min;
    }

    /// <summary>
    /// Calculates the mean of all elements in the source.
    /// </summary>
    /// <param name="source">The array whose mean will be computed.</param>
    /// <returns>The arithmetic mean of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mean(this float[,,,] source) => source.Sum() / source.Length;

    /// <summary>
    /// Calculates the mean of each column in the source.
    /// </summary>
    /// <param name="source">The two-dimensional array to process.</param>
    /// <returns>A one-dimensional array where each element is the mean of a column.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] MeanByColumn(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[] res = new float[columns];

        for (int col = 0; col < columns; col++)
        {
            float sum = 0;
            for (int row = 0; row < rows; row++)
            {
                sum += source[row, col];
            }
            res[col] = sum / rows;
        }

        return res;
    }

    /*
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] MultiplyByTanhDerivative(this float[,,,] outputGradient, float[,,,] output)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Tanh function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Tanh function tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) is 1 - tanh(x)^2.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * (1 - tanh(x)^2).
        // The elementwise multiplication of the output gradient and the derivative of the Tanh function is returned as the input gradient.
        // tanh(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient

        int d0 = outputGradient.GetLength(0);
        int d1 = outputGradient.GetLength(1);
        int d2 = outputGradient.GetLength(2);
        int d3 = outputGradient.GetLength(3);

        Debug.Assert(d0 > 0 && d1 > 0 && d2 > 0 && d3 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(output.GetLength(0) == d0 && output.GetLength(1) == d1 && output.GetLength(2) == d2 && output.GetLength(3) == d3, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,,,] result = new float[d0, d1, d2, d3];

        for (int i = 0; i < d0; i++)
        {
            for (int j = 0; j < d1; j++)
            {
                for (int k = 0; k < d2; k++)
                {
                    for (int l = 0; l < d3; l++)
                    {
                        float y = output[i, j, k, l];
                        float dy = outputGradient[i, j, k, l];
                        float tanhDerivative = 1f - (y * y);
                        result[i, j, k, l] = dy * tanhDerivative;
                    }
                }
            }
        }
        return result;
    }*/

    /// <summary>
    /// Fills the source with random float values between -0.5 and 0.5 using the specified seed.
    /// </summary>
    /// <param name="source">The two-dimensional array to fill with random values.</param>
    /// <param name="seed">The seed used to initialize the random number generator.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void RandomInPlace(this float[,] source, int seed)
    {
        Random rand = new(seed);
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                source[row, col] = rand.NextSingle() - 0.5f;
            }
        }
    }

    /// <summary>
    /// Calculates the standard deviation for all elements of a two-dimensional array.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c> sqrt(sum((x - mean)^2) / N)</c>, where N is the number
    /// of all elements in the array.
    /// </remarks>
    /// <param name="source">The array whose standard deviation will be computed.</param>
    /// <returns>The standard deviation of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float StdDev(this float[,] source)
    {
        Debug.Assert(source.Length > 0, "Source array must contain at least one element to compute standard deviation.");

        float mean = source.Mean();
        float sum = 0;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float value = source[i, j] - mean;
                sum += value * value;
            }
        }

        return MathF.Sqrt(sum / source.Length);
    }

    /// <summary>
    /// Calculates the standard deviation for all elements of a three-dimensional array.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c> sqrt(sum((x - mean)^2) / N)</c>, where N is the number
    /// of all elements in the array.
    /// </remarks>
    /// <param name="source">The array whose standard deviation will be computed.</param>
    /// <returns>The standard deviation of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float StdDev(this float[,,] source)
    {
        Debug.Assert(source.Length > 0, "Source array must contain at least one element to compute standard deviation.");

        float mean = source.Mean();
        float sum = 0;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    float value = source[i, j, k] - mean;
                    sum += value * value;
                }
            }
        }

        return MathF.Sqrt(sum / source.Length);
    }

    /// <summary>
    /// Calculates the standard deviation for all elements of a four-dimensional array.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c> sqrt(sum((x - mean)^2) / N)</c>, where N is the number
    /// of all elements in the array.
    /// </remarks>
    /// <param name="source">The array whose standard deviation will be computed.</param>
    /// <returns>The standard deviation of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float StdDev(this float[,,,] source)
    {
        Debug.Assert(source.Length > 0, "Source array must contain at least one element to compute standard deviation.");

        float mean = source.Mean();
        float sum = 0;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        float value = source[i, j, k, l] - mean;
                        sum += value * value;
                    }
                }
            }
        }

        return MathF.Sqrt(sum / source.Length);
    }

    /// <summary>
    /// Calculates the sum of all elements in the two-dimensional array.
    /// </summary>
    /// <param name="source">The array to sum.</param>
    /// <returns>The sum of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(this float[,] source)
    {
        // Sum over all elements.
        float sum = 0;
        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                sum += source[row, col];
            }
        }

        return sum;
    }

    /// <summary>
    /// Calculates the sum of all elements in the three-dimensional array.
    /// </summary>
    /// <param name="source">The array to sum.</param>
    /// <returns>The sum of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(this float[,,] source)
    {
        // Sum over all elements.
        float sum = 0;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    sum += source[i, j, k];
                }
            }
        }

        return sum;
    }

    /// <summary>
    /// Calculates the sum of all elements in the four-dimensional array.
    /// </summary>
    /// <param name="source">The array to sum.</param>
    /// <returns>The sum of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(this float[,,,] source)
    {
        // Sum over all elements.
        float sum = 0;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        sum += source[i, j, k, l];
                    }
                }
            }
        }

        return sum;
    }

    /// <summary>
    /// Calculates the sum of each column in the source.
    /// </summary>
    /// <param name="source">The two-dimensional array to process.</param>
    /// <returns>A one-dimensional array where each element is the sum of a column.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] SumByColumns(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[] res = new float[columns];

        for (int col = 0; col < columns; col++)
        {
            float sum = 0;
            for (int row = 0; row < rows; row++)
            {
                sum += source[row, col];
            }
            res[col] = sum;
        }

        return res;
    }

   

}