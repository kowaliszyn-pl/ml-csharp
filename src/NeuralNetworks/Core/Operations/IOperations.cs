// Neural Networks in C♯
// File name: IOperations.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Core.Operations;

internal interface IOperations
{
    public float[,,,] Convolve2DForward(float[,,,] input, float[,,,] weights, int? padding = null);
    public float[,,,] Convolve2DBackwardInput(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? padding = null);
    public float[,,,] Convolve2DBackwardWeights(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? padding = null);
    public float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f);
    public float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target);
    public float[,,,] LeakyReLUCalcInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta);
    public float[,,,] MultiplyByTanhDerivative(float[,,,] outputGradient, float[,,,] output);
    public float[,] WeightMultiplyCalcOutput(float[,] input, float[,] weights);
    public float[,] WeightMultiplyCalcInputGradient(float[,] outputGradient, float[,] weights);
    public float[,] WeightMultiplyCalcParamGradient(float[,] input, float[,] outputGradient);
}
