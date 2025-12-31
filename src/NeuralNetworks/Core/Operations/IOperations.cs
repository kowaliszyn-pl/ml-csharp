// Neural Networks in C♯
// File name: IOperations.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Core.Operations;

public interface IOperations
{
    public float[,,,] Convolve2DOutput(float[,,,] input, float[,,,] weights, int? padding = null);
    public float[,,,] Convolve2DInputGradient(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? padding = null);
    public float[,,,] Convolve2DParamGradient(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? padding = null);
    public float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f);
    public float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target);
    public float[,] Flatten(float[,,,] source);
    public float[,,,] LeakyReLU(float[,,,] input, float alpha = 0.01f, float beta = 1f);
    public float[,,,] LeakyReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta);
    public float[,,,] TanhInputGradient(float[,,,] outputGradient, float[,,,] output);
    public float[,,,] Tanh(float[,,,] source);
    public float[,,,] Unflatten(float[,] source, float[,,,] targetSize);
    public float[,] WeightMultiplyOutput(float[,] input, float[,] weights);
    public float[,] WeightMultiplyInputGradient(float[,] outputGradient, float[,] weights);
    public float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient);

    OperationBackendType BackendType { get; }
}
