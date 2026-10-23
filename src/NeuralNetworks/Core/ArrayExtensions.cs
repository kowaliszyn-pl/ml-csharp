// Neural Networks in C♯
// File name: ArrayExtensions.cs
// www.kowaliszyn.pl, 2025 - 2026

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





}