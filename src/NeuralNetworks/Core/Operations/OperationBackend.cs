// Neural Networks in C♯
// File name: OperationBackend.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

using ILGPU.Runtime.Cuda;

namespace NeuralNetworks.Core.Operations;

public static class OperationBackend
{

    #region Backend Management

    private static bool s_statisticsEnabled = false;

    private static long s_convolve2DOutputTicks;
    private static long s_convolve2DOutputCalls;
    private static long s_convolve2DOutputInputMemory;

    private static long s_convolve2DInputGradientTicks;
    private static long s_convolve2DInputGradientCalls;
    private static long s_convolve2DInputGradientInputMemory;

    private static long s_convolve2DParamGradientTicks;
    private static long s_convolve2DParamGradientCalls;
    private static long s_convolve2DParamGradientInputMemory;

    private static long s_weightMultiplyOutputTicks;
    private static long s_weightMultiplyOutputCalls;
    private static long s_weightMultiplyOutputInputMemory;

    private static long s_weightMultiplyInputGradientTicks;
    private static long s_weightMultiplyInputGradientCalls;
    private static long s_weightMultiplyInputGradientInputMemory;

    private static long s_weightMultiplyParamGradientTicks;
    private static long s_weightMultiplyParamGradientCalls;
    private static long s_weightMultiplyParamGradientInputMemory;

    static OperationBackend()
    {
        AppDomain.CurrentDomain.ProcessExit += (s, e) => DisposeCurrentOperationBackend();
    }

    public static OperationBackendType CurrentType => Current.BackendType;

    public static bool StatisticsEnabled
    {
        get
        {
            return s_statisticsEnabled;
        }
        set
        {
            if (s_statisticsEnabled != value)
            {
                s_statisticsEnabled = value;
                if (s_statisticsEnabled)
                {
                    ResetStatistics();
                }
            }
        }
    }

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

    internal static string GetStatistics()
    {
        if (!s_statisticsEnabled)
        {
            return "Timing disabled.";
        }

        return string.Join(
            Environment.NewLine,
            Format("Convolve2DOutput", s_convolve2DOutputTicks, s_convolve2DOutputCalls, s_convolve2DOutputInputMemory),
            Format("Convolve2DInputGradient", s_convolve2DInputGradientTicks, s_convolve2DInputGradientCalls, s_convolve2DInputGradientInputMemory),
            Format("Convolve2DParamGradient", s_convolve2DParamGradientTicks, s_convolve2DParamGradientCalls, s_convolve2DParamGradientInputMemory),
            Format("WeightMultiplyOutput", s_weightMultiplyOutputTicks, s_weightMultiplyOutputCalls, s_weightMultiplyOutputInputMemory),
            Format("WeightMultiplyInputGradient", s_weightMultiplyInputGradientTicks, s_weightMultiplyInputGradientCalls, s_weightMultiplyInputGradientInputMemory),
            Format("WeightMultiplyParamGradient", s_weightMultiplyParamGradientTicks, s_weightMultiplyParamGradientCalls, s_weightMultiplyParamGradientInputMemory)
        );
    }

    internal static void ResetStatistics()
    {
        s_convolve2DOutputTicks = 0;
        s_convolve2DOutputCalls = 0;
        s_convolve2DOutputInputMemory = 0;

        s_convolve2DInputGradientTicks = 0;
        s_convolve2DInputGradientCalls = 0;
        s_convolve2DInputGradientInputMemory = 0;

        s_convolve2DParamGradientTicks = 0;
        s_convolve2DParamGradientCalls = 0;
        s_convolve2DParamGradientInputMemory = 0;

        s_weightMultiplyOutputTicks = 0;
        s_weightMultiplyOutputCalls = 0;
        s_weightMultiplyOutputInputMemory = 0;

        s_weightMultiplyInputGradientTicks = 0;
        s_weightMultiplyInputGradientCalls = 0;
        s_weightMultiplyInputGradientInputMemory = 0;

        s_weightMultiplyParamGradientTicks = 0;
        s_weightMultiplyParamGradientCalls = 0;
        s_weightMultiplyParamGradientInputMemory = 0;
    }

    private static string Format(string name, long ticks, long calls, long memoryBytes)
    {
        double totalSeconds = (double)ticks / Stopwatch.Frequency;
        double averageMicroseconds = calls > 0 ? (totalSeconds * 1000.0) / calls : 0.0;
        double memoryMB = (double)memoryBytes / (1024 * 1024);
        double memoryMBPerCall = calls > 0 ? memoryMB / calls : 0.0;
        return $"{name}: calls={calls}, totalSeconds={totalSeconds:F2}, averageMicroseconds={averageMicroseconds:F3}, inputMemoryMB={memoryMB:F2}, memoryMBPerCall={memoryMBPerCall:F4}";
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
    internal static float[,] BipolarSigmoidOutput(float[,] input, float scale)
        => Current.BipolarSigmoidOutput(input, scale);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] BipolarSigmoidInputGradient(float[,] outputGradient, float[,] output, float scale)
        => Current.BipolarSigmoidInputGradient(outputGradient, output, scale);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] LeakyReLUOutput(float[,,,] input, float alpha = 0.01f, float beta = 1f)
        => Current.LeakyReLUOutput(input, alpha, beta);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] LeakyReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta)
        => Current.LeakyReLUInputGradient(outputGradient, input, alfa, beta);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] LeakyReLUOutput(float[,] input, float alpha = 0.01f, float beta = 1f)
       => Current.LeakyReLUOutput(input, alpha, beta);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] LeakyReLUInputGradient(float[,] outputGradient, float[,] input, float alfa, float beta)
        => Current.LeakyReLUInputGradient(outputGradient, input, alfa, beta);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] ReLUOutput(float[,] input, float beta = 1f)
        => Current.ReLUOutput(input, beta);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] ReLUInputGradient(float[,] outputGradient, float[,] input, float beta)
        => Current.ReLUInputGradient(outputGradient, input, beta);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] ReLUOutput(float[,,,] input, float beta = 1f)
        => Current.ReLUOutput(input, beta);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] ReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float beta)
        => Current.ReLUInputGradient(outputGradient, input, beta);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] SigmoidOutput(float[,] input)
        => Current.SigmoidOutput(input);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] SigmoidInputGradient(float[,] outputGradient, float[,] output)
        => Current.SigmoidInputGradient(outputGradient, output);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] SoftplusOutput(float[,] input)
        => Current.SoftplusOutput(input);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] SoftplusInputGradient(float[,] outputGradient, float[,] output)
        => Current.SoftplusInputGradient(outputGradient, output);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] TanhOutput(float[,] input)
        => Current.TanhOutput(input);

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the input.
    /// </summary>
    /// <returns>A new input with the hyperbolic tangent applied element-wise.</returns>
    /// <param name="input">The four-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] TanhOutput(float[,,,] input)
        => Current.TanhOutput(input);

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
    internal static float[,] TanhInputGradient(float[,] outputGradient, float[,] output)
      => Current.TanhInputGradient(outputGradient, output);

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

    #region Bias Addition Operations

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] BiasAddOutput(float[,] input, float[] bias)
        => Current.BiasAddOutput(input, bias);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[] BiasAddParamGradient(float[,] outputGradient)
        // outputGradient is already averaged over the batch size in the loss function, so we just need to sum by columns
        => Current.BiasAddParamGradient(outputGradient);

    #endregion

    #region Bias Addition Conv1D Operations

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,] BiasAddConv1DOutput(float[,,] input, float[] bias)
        => Current.BiasAddConv1DOutput(input, bias);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[] BiasAddConv1DParamGradient(float[,,] outputGradient)
        // outputGradient is already averaged over the batch size in the loss function, so we just need to sum by columns
        => Current.BiasAddConv1DParamGradient(outputGradient);

    #endregion

    #region Convolution 2D Operations

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
    internal static float[,,,] Convolve2DOutput(float[,,,] input, float[,,,] weights, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 0, int dilatationWidth = 0)
    {
        long start = s_statisticsEnabled ? Stopwatch.GetTimestamp() : 0;

        float[,,,] result = Current.Convolve2DOutput(input, weights, paddingHeight, paddingWidth, strideHeight, strideWidth, dilatationHeight, dilatationWidth);

        if (s_statisticsEnabled)
        {
            long elapsed = Stopwatch.GetTimestamp() - start;
            Interlocked.Add(ref s_convolve2DOutputTicks, elapsed);
            Interlocked.Increment(ref s_convolve2DOutputCalls);
            Interlocked.Add(ref s_convolve2DOutputInputMemory, (input.Length + weights.Length) * sizeof(float));
        }

        return result;
    }

    /// <summary>
    /// Computes the gradient of the input array for a 2D convolution operation during backpropagation.
    /// </summary>
    /// <param name="input">The input array to the convolution layer, represented as a four-dimensional array with shape [batchSize, inputChannels, inputHeight, inputWidth].</param>
    /// <param name="weights">The weights of the convolution filters, represented as a four-dimensional array with shape [inputChannels, outputChannels, kernelHeight, kernelWidth].</param>
    /// <param name="outputGradient">The gradient of the loss with respect to the output of the convolution layer, represented as a four-dimensional array with shape [batchSize, outputChannels, outputHeight, outputWidth].</param>
    /// <returns>The input gradient array of shape [batchSize, inputChannels, inputHeight, inputWidth].
    /// </returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] Convolve2DInputGradient(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 0, int dilatationWidth = 0)
    {
        long start = s_statisticsEnabled ? Stopwatch.GetTimestamp() : 0;

        float[,,,] result = Current.Convolve2DInputGradient(input, weights, outputGradient, paddingHeight, paddingWidth, strideHeight, strideWidth, dilatationHeight, dilatationWidth);

        if (s_statisticsEnabled)
        {
            long elapsed = Stopwatch.GetTimestamp() - start;
            Interlocked.Add(ref s_convolve2DInputGradientTicks, elapsed);
            Interlocked.Increment(ref s_convolve2DInputGradientCalls);
            Interlocked.Add(ref s_convolve2DInputGradientInputMemory, (input.Length + weights.Length + outputGradient.Length) * sizeof(float));
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,,,] Convolve2DParamGradient(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 0, int dilatationWidth = 0)
    {
        long start = s_statisticsEnabled ? Stopwatch.GetTimestamp() : 0;

        float[,,,] result = Current.Convolve2DParamGradient(input, outputGradient, kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilatationHeight, dilatationWidth);

        if (s_statisticsEnabled)
        {
            long elapsed = Stopwatch.GetTimestamp() - start;
            Interlocked.Add(ref s_convolve2DParamGradientTicks, elapsed);
            Interlocked.Increment(ref s_convolve2DParamGradientCalls);
            Interlocked.Add(ref s_convolve2DParamGradientInputMemory, (input.Length + outputGradient.Length) * sizeof(float));
        }

        return result;
    }

    #endregion

    #region Convolution 1D Operations

    internal static float[,,] Convolve1DOutput(float[,,] input, float[,,] weights, int padding, int stride, int dilatation)
        => Current.Convolve1DOutput(input, weights, padding, stride, dilatation);

    internal static float[,,] Convolve1DInputGradient(float[,,] input, float[,,] weights, float[,,] outputGradient, int padding, int stride, int dilatation)
        => Current.Convolve1DInputGradient(input, weights, outputGradient, padding, stride, dilatation);

    internal static float[,,] Convolve1DParamGradient(float[,,] input, float[,,] outputGradient, int padding, int stride, int dilatation)
        => Current.Convolve1DParamGradient(input, outputGradient, padding, stride, dilatation);

    #endregion

    #region Weight Multiplication Operations

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] WeightMultiplyOutput(float[,] input, float[,] weights)
    {
        long start = s_statisticsEnabled ? Stopwatch.GetTimestamp() : 0;

        float[,] result = Current.WeightMultiplyOutput(input, weights);

        if (s_statisticsEnabled)
        {
            long elapsed = Stopwatch.GetTimestamp() - start;
            Interlocked.Add(ref s_weightMultiplyOutputTicks, elapsed);
            Interlocked.Increment(ref s_weightMultiplyOutputCalls);
            Interlocked.Add(ref s_weightMultiplyOutputInputMemory, (input.Length + weights.Length) * sizeof(float));
        }

        return result;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] WeightMultiplyInputGradient(float[,] outputGradient, float[,] weights)
    {
        long start = s_statisticsEnabled ? Stopwatch.GetTimestamp() : 0;

        float[,] result = Current.WeightMultiplyInputGradient(outputGradient, weights);

        if (s_statisticsEnabled)
        {
            long elapsed = Stopwatch.GetTimestamp() - start;
            Interlocked.Add(ref s_weightMultiplyInputGradientTicks, elapsed);
            Interlocked.Increment(ref s_weightMultiplyInputGradientCalls);
            Interlocked.Add(ref s_weightMultiplyInputGradientInputMemory, (outputGradient.Length + weights.Length) * sizeof(float));
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient)
    {
        long start = s_statisticsEnabled ? Stopwatch.GetTimestamp() : 0;

        float[,] result = Current.WeightMultiplyParamGradient(input, outputGradient);

        if (s_statisticsEnabled)
        {
            long elapsed = Stopwatch.GetTimestamp() - start;
            Interlocked.Add(ref s_weightMultiplyParamGradientTicks, elapsed);
            Interlocked.Increment(ref s_weightMultiplyParamGradientCalls);
            Interlocked.Add(ref s_weightMultiplyParamGradientInputMemory, (input.Length + outputGradient.Length) * sizeof(float));
        }

        return result;
    }

    #endregion

    #endregion

    #region Dropout

    internal static float[,] DropoutOutput(float[,] input, bool inference, float keepProb, SeededRandom? random, out float[,]? mask)
        => Current.DropoutOutput(input, inference, keepProb, random, out mask);

    internal static float[,] DropoutInputGradient(float[,] outputGradient, float[,] mask)
        => Current.DropoutInputGradient(outputGradient, mask);

    internal static float[,,,] DropoutOutput(float[,,,] input, bool inference, float keepProb, SeededRandom? random, out float[,,,]? mask)
        => Current.DropoutOutput(input, inference, keepProb, random, out mask);

    internal static float[,,,] DropoutInputGradient(float[,,,] outputGradient, float[,,,] mask)
        => Current.DropoutInputGradient(outputGradient, mask);

    internal static float[,] InvertedDropoutOutput(float[,] input, bool inference, float keepProb, SeededRandom? random, out float[,]? mask)
        => Current.InvertedDropoutOutput(input, inference, keepProb, random, out mask);

    internal static float[,] InvertedDropoutInputGradient(float[,] outputGradient, float[,] mask, float keepProb)
        => Current.InvertedDropoutInputGradient(outputGradient, mask, keepProb);

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
