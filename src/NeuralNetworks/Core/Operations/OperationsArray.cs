// Neural Networks in C♯
// File name: OperationsArray.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Operations;

internal class OperationsArray : IOperations
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        // Clip the probabilities to avoid log(0).
        float[,] clippedSoftmax = predicted.Clip(eps, 1 - eps);
        return -clippedSoftmax.Log().MultiplyElementwise(target).Mean();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        int batchSize = predicted.GetLength(0);
        return predicted.Subtract(target).Divide(batchSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float[,] WeightMultiplyCalcOutput(float[,] input, float[,] weights)
        => input.MultiplyDot(weights);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float[,] WeightMultiplyCalcInputGradient(float[,] outputGradient, float[,] weights)
        => outputGradient.MultiplyDot(weights.Transpose());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float[,] WeightMultiplyCalcParamGradient(float[,] input, float[,] outputGradient)
        => input.Transpose().MultiplyDot(outputGradient);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float[,,,] Convolve2DForward(float[,,,] input, float[,,,] weights, int? paddingArg = null)
    {
        int batchSize = input.GetLength(0);
        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputChannels = weights.GetLength(1);
        int kernelSize = weights.GetLength(2);

        Debug.Assert(weights.GetLength(0) == inputChannels);
        Debug.Assert(kernelSize == weights.GetLength(3));

        int padding = paddingArg ?? kernelSize / 2;

        int outputHeight = inputHeight - kernelSize + 1 + 2 * padding;
        int outputWidth = inputWidth - kernelSize + 1 + 2 * padding;

        float[,,,] output = new float[batchSize, outputChannels, outputHeight, outputWidth];

        for (int b = 0; b < batchSize; b++)
        {
            for (int oc = 0; oc < outputChannels; oc++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        float sum = 0.0f;
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            for (int kh = 0; kh < kernelSize; kh++)
                            {
                                for (int kw = 0; kw < kernelSize; kw++)
                                {
                                    int ih = oh + kh - padding;
                                    int iw = ow + kw - padding;
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                    {
                                        sum += input[b, ic, ih, iw] * weights[ic, oc, kh, kw];
                                    }
                                }
                            }
                        }
                        output[b, oc, oh, ow] = sum;
                    }
                }
            }
        }

        return output;

    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float[,,,] Convolve2DBackwardInput(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? paddingArg = null)
    {
        int batchSize = outputGradient.GetLength(0);
        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputChannels = outputGradient.GetLength(1);
        int kernelSize = weights.GetLength(2);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        Debug.Assert(weights.GetLength(0) == inputChannels);
        Debug.Assert(kernelSize == weights.GetLength(3));

        int padding = paddingArg ?? kernelSize / 2;

        float[,,,] inputGradient = new float[batchSize, inputChannels, inputHeight, inputWidth];

        for (int b = 0; b < batchSize; b++)
        {
            for (int ic = 0; ic < inputChannels; ic++)
            {
                for (int ih = 0; ih < inputHeight; ih++)
                {
                    for (int iw = 0; iw < inputWidth; iw++)
                    {
                        float sum = 0.0f;
                        for (int oc = 0; oc < outputChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelSize; kh++)
                            {
                                for (int kw = 0; kw < kernelSize; kw++)
                                {
                                    int oh = ih - kh + padding;
                                    int ow = iw - kw + padding;
                                    if (oh >= 0 && oh < outputGradientHeight && ow >= 0 && ow < outputGradientWidth)
                                    {
                                        sum += outputGradient[b, oc, oh, ow] * weights[ic, oc, kh, kw];
                                    }
                                }
                            }
                        }
                        inputGradient[b, ic, ih, iw] = sum;
                    }
                }
            }
        }

        return inputGradient;

    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float[,,,] Convolve2DBackwardWeights(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? paddingArg = null)
    {
        int batchSize = outputGradient.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        Debug.Assert(kernelHeight == kernelWidth);

        int padding = paddingArg ?? kernelHeight / 2;

        float[,,,] paramGradient = new float[inputChannels, outputChannels, kernelHeight, kernelWidth];

        for (int b = 0; b < batchSize; b++)
        {
            for (int ic = 0; ic < inputChannels; ic++)
            {
                for (int oc = 0; oc < outputChannels; oc++)
                {
                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            float sum = 0.0f;
                            for (int oh = 0; oh < outputGradientHeight; oh++)
                            {
                                for (int ow = 0; ow < outputGradientWidth; ow++)
                                {
                                    int ih = oh + kh - padding;
                                    int iw = ow + kw - padding;
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                    {
                                        sum += outputGradient[b, oc, oh, ow] * input[b, ic, ih, iw];
                                    }
                                }
                            }
                            paramGradient[ic, oc, kh, kw] += sum;
                        }
                    }
                }
            }
        }

        return paramGradient;

    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float[,,,] LeakyReLUCalcInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);
        int dim3 = input.GetLength(2);
        int dim4 = input.GetLength(3);
        float[,,,] inputGradient = new float[dim1, dim2, dim3, dim4];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        inputGradient[i, j, k, l] = input[i, j, k, l] > 0 ? outputGradient[i, j, k, l] * beta : outputGradient[i, j, k, l] * alfa;
                    }
                }
            }
        }
        return inputGradient;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public virtual float[,,,] MultiplyByTanhDerivative(float[,,,] outputGradient, float[,,,] output)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Tanh function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Tanh function tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) is 1 - tanh(x)^2.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * (1 - tanh(x)^2).
        // The elementwise multiplication of the output gradient and the derivative of the Tanh function is returned as the input gradient.
        // tanh(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient
        float[,,,] tanhBackward = output.AsOnes().Subtract(output.MultiplyElementwise(output));
        return outputGradient.MultiplyElementwise(tanhBackward);
    }
}
