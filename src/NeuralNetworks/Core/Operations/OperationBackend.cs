// Neural Networks in C♯
// File name: OperationBackend.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Core.Operations;

public static class OperationBackend
{

    static OperationBackend()
    {
        AppDomain.CurrentDomain.ProcessExit += (s, e) => DisposeCurrentOperationBackend();
    }

    public static OperationBackendType CurrentType
    {
        get
        {
            if (Current is OperationsGpu)
            {
                return OperationBackendType.Gpu;
            }
            else if (Current is OperationsSpanParallel)
            {
                return OperationBackendType.Cpu_Spans_Parallel;
            }
            else if (Current is OperationsSpan)
            {
                return OperationBackendType.Cpu_Spans;
            }
            else if (Current is OperationsArray)
            {
                return OperationBackendType.Cpu_Arrays;
            }
            else
            {
                throw new NotSupportedException("The current operation backend type is not valid.");
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
            OperationBackendType.Cpu_Arrays => new OperationsArray(),
            OperationBackendType.Cpu_Spans => new OperationsSpan(),
            OperationBackendType.Cpu_Spans_Parallel => new OperationsSpanParallel(),
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

    internal static float[,,,] Convolve2DForward(float[,,,] input, float[,,,] weights, int? padding = null)
        => Current.Convolve2DForward(input, weights, padding);

    internal static float[,,,] Convolve2DBackwardInput(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? padding = null)
        => Current.Convolve2DBackwardInput(input, weights, outputGradient, padding);

    internal static float[,,,] Convolve2DBackwardWeights(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? padding = null)
        => Current.Convolve2DBackwardWeights(input, outputGradient, kernelHeight, kernelWidth, padding);

    internal static float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
        => Current.CrossEntropyLoss(predicted, target, eps);

    internal static float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
        => Current.CrossEntropyLossGradient(predicted, target);

    internal static float[,] Flatten(float[,,,] source)
        => Current.Flatten(source);

    internal static float[,,,] LeakyReLU(float[,,,] input, float alpha = 0.01f, float beta = 1f)
        => Current.LeakyReLU(input, alpha, beta);

    internal static float[,,,] LeakyReLUCalcInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta)
        => Current.LeakyReLUCalcInputGradient(outputGradient, input, alfa, beta);

    internal static float[,,,] MultiplyByTanhDerivative(float[,,,] outputGradient, float[,,,] output)
        => Current.MultiplyByTanhDerivative(outputGradient, output);

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the source.
    /// </summary>
    /// <returns>A new source with the hyperbolic tangent applied element-wise.</returns>
    /// <param name="source">The four-dimensional array to transform.</param>
    internal static float[,,,] Tanh(float[,,,] source)
        => Current.Tanh(source);

    internal static float[,,,] Unflatten(float[,] source, float[,,,] targetSize) 
        => Current.Unflatten(source, targetSize);

    internal static float[,] WeightMultiplyCalcOutput(float[,] input, float[,] weights)
        => Current.WeightMultiplyCalcOutput(input, weights);

    internal static float[,] WeightMultiplyCalcInputGradient(float[,] outputGradient, float[,] weights)
        => Current.WeightMultiplyCalcInputGradient(outputGradient, weights);

    internal static float[,] WeightMultiplyCalcParamGradient(float[,] input, float[,] outputGradient)
        => Current.WeightMultiplyCalcParamGradient(input, outputGradient);
}
