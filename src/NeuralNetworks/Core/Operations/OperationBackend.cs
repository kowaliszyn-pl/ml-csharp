// Neural Networks in C♯
// File name: OperationBackend.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Core.Operations;

internal static class OperationBackend
{

    public static OperationBackendType CurrentType
    {
        get;
        private set;
    }

    public static IOperations Current
    {
        get;
        private set;
    } = null!;

    public static void Use(OperationBackendType backendType)
    {
        CurrentType = backendType;
        if(Current != null)
        {
            if (Current is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }

        Current = backendType switch
        {
            OperationBackendType.Cpu_Arrays => new OperationsArray(),
            OperationBackendType.Cpu_Spans => new OperationsSpan(),
            OperationBackendType.Gpu => new OperationsGpu(),
            _ => throw new NotSupportedException($"The specified backend type '{backendType}' is not supported."),
        };
    }

    public static float[,,,] Convolve2DForward(float[,,,] input, float[,,,] weights, int? padding = null)
        => Current.Convolve2DForward(input, weights, padding);

    public static float[,,,] Convolve2DBackwardInput(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? padding = null)
        => Current.Convolve2DBackwardInput(input, weights, outputGradient, padding);

    public static float[,,,] Convolve2DBackwardWeights(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? padding = null)
        => Current.Convolve2DBackwardWeights(input, outputGradient, kernelHeight, kernelWidth, padding);

    public static float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
        => Current.CrossEntropyLoss(predicted, target, eps);

    public static float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
        => Current.CrossEntropyLossGradient(predicted, target);

    public static float[,,,] LeakyReLUCalcInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta)
        => Current.LeakyReLUCalcInputGradient(outputGradient, input, alfa, beta);

    public static float[,,,] MultiplyByTanhDerivative(float[,,,] outputGradient, float[,,,] output)
        => Current.MultiplyByTanhDerivative(outputGradient, output);

    public static float[,] WeightMultiplyCalcOutput(float[,] input, float[,] weights)
        => Current.WeightMultiplyCalcOutput(input, weights);

    public static float[,] WeightMultiplyCalcInputGradient(float[,] outputGradient, float[,] weights)
        => Current.WeightMultiplyCalcInputGradient(outputGradient, weights);

    public static float[,] WeightMultiplyCalcParamGradient(float[,] input, float[,] outputGradient)
        => Current.WeightMultiplyCalcParamGradient(input, outputGradient);
}
