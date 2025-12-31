// Neural Networks in C♯
// File name: OperationBackend.cs
// www.kowaliszyn.pl, 2025

using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Operations;

public static class OperationBackend
{

    #region Backend Management

    static OperationBackend()
    {
        AppDomain.CurrentDomain.ProcessExit += (s, e) => DisposeCurrentOperationBackend();
    }

    public static OperationBackendType CurrentType => Current.BackendType;


    internal static IOperations Current
    {
        get;
        private set;
    } = new OperationsArray();

    public static void Use(OperationBackendType backendType)
    {
        DisposeCurrentOperationBackend();

        Current = backendType switch
        {
            OperationBackendType.CpuArrays => new OperationsArray(),
            OperationBackendType.CpuSpans => new OperationsSpan(),
            OperationBackendType.CpuSpansParallel => new OperationsSpanParallel(),
            OperationBackendType.Gpu => new OperationsGpu(),
            _ => throw new NotSupportedException($"The specified backend type '{backendType}' is not supported."),
        };
    }

    private static void DisposeCurrentOperationBackend()
    {
        if (Current != null)
        {
            if (Current is IDisposable disposable)
            {
                disposable.Dispose();
                Console.WriteLine("Disposed current operation backend.");
            }
            Current = null!;
        }
    }

    #endregion

    #region Loss Functions

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
       => Current.CrossEntropyLoss(predicted, target, eps);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
        => Current.CrossEntropyLossGradient(predicted, target);

    #endregion

    #region Activation Functions

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] LeakyReLU(float[,,,] input, float alpha = 0.01f, float beta = 1f)
        => Current.LeakyReLU(input, alpha, beta);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] LeakyReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta)
        => Current.LeakyReLUInputGradient(outputGradient, input, alfa, beta);

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the input.
    /// </summary>
    /// <returns>A new input with the hyperbolic tangent applied element-wise.</returns>
    /// <param name="input">The four-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] Tanh(float[,,,] input)
        => Current.Tanh(input);

    /// <summary>
    /// Calculates the gradient of the loss with respect to the input of the Tanh activation function.
    /// </summary>
    /// <remarks>
    /// Given the output gradient (dL/dy), the function calculates the source gradient (dL/dx). 
    /// <para/>
    /// The derivative of the Tanh function <c>tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))</c> is <c>1 - tanh(x)^2</c>.
    /// Therefore, the source gradient is computed as: <c>dL/dx = dL/dy * (1 - tanh(x)^2) = dL/dy * (1 - output^2)</c>.
    /// <list type="bullet">
    /// <item>
    /// tanh(x) => output
    /// </item>
    /// <item>
    /// dL/dy => outputGradient
    /// </item>
    /// <item>
    /// dL/dx => inputGradient
    /// </item>
    /// </list>
    /// </remarks>
    /// <param name="output">The output of the Tanh function (<c>tanh(x)</c>).</param>
    /// <param name="outputGradient">The gradient of the loss with respect to the output of the Tanh function (dL/dy).</param>
    /// <returns>
    /// The gradient of the loss with respect to the input of the Tanh function (dL/dx), having the same shape as <paramref name="outputGradient"/>.
    /// </returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] TanhInputGradient(float[,,,] outputGradient, float[,,,] output)
       => Current.TanhInputGradient(outputGradient, output);

    #endregion

    #region Parametric Operations

    // Convolution Operations

    /// <summary>
    /// Computes the 2D convolution of the input array with the specified weights.
    /// </summary>
    /// <remarks>
    /// Padding is symmetric and computed as kernelSize / 2. Strides and dilation are not supported yet.
    /// </remarks>
    /// <param name="input">The input array of shape [batchSize, inputChannels, inputHeight, inputWidth]</param> 
    /// <param name="weights">The weights array (of the convolution filters) of shape [inputChannels, outputChannels, kernelHeight, kernelWidth]</param>
    /// <returns>The output array of shape [batchSize, outputChannels, outputHeight, outputWidth]</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] Convolve2DOutput(float[,,,] input, float[,,,] weights)
        => Current.Convolve2DOutput(input, weights);

    /// <summary>
    /// Computes the gradient of the input array for a 2D convolution operation during backpropagation.
    /// </summary>
    /// <param name="input">The input array to the convolution layer, represented as a four-dimensional array with shape [batchSize, inputChannels, inputHeight, inputWidth].</param>
    /// <param name="weights">The weights of the convolution filters, represented as a four-dimensional array with shape [inputChannels, outputChannels, kernelHeight, kernelWidth].</param>
    /// <param name="outputGradient">The gradient of the loss with respect to the output of the convolution layer, represented as a four-dimensional array with shape [batchSize, outputChannels, outputHeight, outputWidth].</param>
    /// <returns>The input gradient array of shape [batchSize, inputChannels, inputHeight, inputWidth].
    /// </returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] Convolve2DInputGradient(float[,,,] input, float[,,,] weights, float[,,,] outputGradient)
        => Current.Convolve2DInputGradient(input, weights, outputGradient);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] Convolve2DParamGradient(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth)
        => Current.Convolve2DParamGradient(input, outputGradient, kernelHeight, kernelWidth);

    // Weight Multiplication Operations

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] WeightMultiplyOutput(float[,] input, float[,] weights)
        => Current.WeightMultiplyOutput(input, weights);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] WeightMultiplyInputGradient(float[,] outputGradient, float[,] weights)
        => Current.WeightMultiplyInputGradient(outputGradient, weights);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient)
        => Current.WeightMultiplyParamGradient(input, outputGradient);

    #endregion

    #region Transformations

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] Flatten(float[,,,] source)
        => Current.Flatten(source);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] Unflatten(float[,] source, float[,,,] targetSize) 
        => Current.Unflatten(source, targetSize);

    #endregion
}
