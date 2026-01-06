// Neural Networks in C♯
// File name: OperationsArray.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core.Extensions;

namespace NeuralNetworks.Core.Operations;

/// <summary>
/// Provides the baseline array-based CPU implementation of <see cref="IOperations"/> for deterministic reference execution.
/// </summary>
/// <remarks>
/// All kernels are written against standard multidimensional arrays and straightforward loops so the behavior mirrors
/// higher-performance backends while remaining easy to debug, test, and teach. Use this backend when GPU acceleration
/// is unavailable, when numerical traceability matters more than throughput, or as a correctness oracle for other
/// implementations.
/// </remarks>
internal class OperationsArray : IOperations
{
    #region Backend Management

    public virtual OperationBackendType BackendType => OperationBackendType.CpuArrays;

    #endregion

    #region Loss Functions

    public virtual float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        // Clip the probabilities to avoid log(0).
        float[,] clippedSoftmax = predicted.Clip(eps, 1 - eps);
        return -clippedSoftmax.Log().MultiplyElementwise(target).Mean();
    }

    public virtual float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        int batchSize = predicted.GetLength(0);
        return predicted.Subtract(target).Divide(batchSize);
    }

    #endregion

    #region Activations Functions

    public virtual float[,,,] LeakyReLU(float[,,,] input, float alpha = 0.01F, float beta = 1)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);
        int dim3 = input.GetLength(2);
        int dim4 = input.GetLength(3);
        float[,,,] output = new float[dim1, dim2, dim3, dim4];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        output[i, j, k, l] = input[i, j, k, l] > 0 ? input[i, j, k, l] * beta : input[i, j, k, l] * alpha;
                    }
                }
            }
        }
        return output;
    }

    public virtual float[,,,] LeakyReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta)
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

    public virtual float[,] Tanh(float[,] input) 
        => input.Tanh();

    public virtual float[,] TanhInputGradient(float[,] outputGradient, float[,] output)
    {
        // The TanhInputGradient function computes the gradient of the loss with respect to the input of the Tanh function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Tanh function tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) is 1 - tanh(x)^2.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * (1 - tanh(x)^2).
        // The elementwise multiplication of the output gradient and the derivative of the Tanh function is returned as the input gradient.
        // tanh(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient
        float[,] tanhBackward = output.AsOnes().Subtract(output.MultiplyElementwise(output));
        return outputGradient.MultiplyElementwise(tanhBackward);
    }

    public virtual float[,,,] Tanh(float[,,,] input) 
        => input.Tanh();

    public virtual float[,,,] TanhInputGradient(float[,,,] outputGradient, float[,,,] output)
    {
        // The TanhInputGradient function computes the gradient of the loss with respect to the input of the Tanh function.
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

    #endregion

    #region Parametric Operations

    // Bias Addition Operations

    public virtual float[,] BiasAddOutput(float[,] input, float[] bias)
        => input.AddRow(bias);

    public virtual float[] BiasAddParamGradient(float[,] outputGradient)
        => outputGradient.SumByColumns();

    // Convolution Operations

    public virtual float[,,,] Convolve2DOutput(float[,,,] input, float[,,,] weights)
    {
        int batchSize = input.GetLength(0);
        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputChannels = weights.GetLength(1);
        int kernelSize = weights.GetLength(2);

        Debug.Assert(weights.GetLength(0) == inputChannels);
        Debug.Assert(kernelSize == weights.GetLength(3));

        int padding = kernelSize / 2;

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

    public virtual float[,,,] Convolve2DInputGradient(float[,,,] input, float[,,,] weights, float[,,,] outputGradient)
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

        int padding = kernelSize / 2;

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

    public virtual float[,,,] Convolve2DParamGradient(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth)
    {
        int batchSize = outputGradient.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        Debug.Assert(kernelHeight == kernelWidth);

        int padding = kernelHeight / 2;

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

    // Weight Multiplication Operations

    public virtual float[,] WeightMultiplyOutput(float[,] input, float[,] weights)
    => input.MultiplyDot(weights);

    public virtual float[,] WeightMultiplyInputGradient(float[,] outputGradient, float[,] weights)
        => outputGradient.MultiplyDot(weights.Transpose());

    public virtual float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient)
        => input.Transpose().MultiplyDot(outputGradient);

    #endregion

    #region Dropout

    public virtual float[,] DropoutOutput(float[,] input, bool inference, float keepProb, SeededRandom? random, out float[,]? mask)
    {
        if (inference)
        {
            mask = null;
            return input.Multiply(keepProb);
        }
        else
        {
            mask = input.AsZeroOnes(keepProb, random ?? new());
            return input.MultiplyElementwise(mask);
        }
    }

    public float[,] DropoutInputGradient(float[,] outputGradient, float[,] mask)
        => outputGradient.MultiplyElementwise(mask);

    public virtual float[,,,] DropoutOutput(float[,,,] input, bool inference, float keepProb, SeededRandom? random, out float[,,,]? mask)
    {
        if (inference)
        {
            mask = null;
            return input.Multiply(keepProb);
        }
        else
        {
            mask = input.AsZeroOnes(keepProb, random ?? new());
            return input.MultiplyElementwise(mask);
        }
    }

    public float[,,,] DropoutInputGradient(float[,,,] outputGradient, float[,,,] mask)
        => outputGradient.MultiplyElementwise(mask);

    public virtual float[,] InvertedDropoutOutput(float[,] input, bool inference, float keepProb, SeededRandom? random, out float[,]? mask)
    {
        if (inference)
        {
            mask = null;
            return input;
        }
        else
        {
            float multiplier = 1f / keepProb;
            mask = input.AsZeroOnes(keepProb, random ?? new());
            return input.MultiplyElementwise(mask).Multiply(multiplier);
        }
    }

    public float[,] InvertedDropoutInputGradient(float[,] outputGradient, float[,] mask, float keepProb)
    {
        float multiplier = 1f / keepProb;
        return outputGradient.MultiplyElementwise(mask).Multiply(multiplier);
    }

    #endregion

    #region Transformations

    public virtual float[,] Flatten(float[,,,] source)
    {
        // Flattent the source for each batch

        int batchSize = source.GetLength(0);
        int channels = source.GetLength(1);
        int height = source.GetLength(2);
        int width = source.GetLength(3);

        float[,] res = new float[batchSize, channels * height * width];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int index = c * height * width + h * width + w;
                        res[b, index] = source[b, c, h, w];
                    }
                }
            }
        }

        return res;
    }

    public virtual float[,,,] Unflatten(float[,] source, float[,,,] targetSize)
    {
        int batchSize = targetSize.GetLength(0);
        int channels = targetSize.GetLength(1);
        int height = targetSize.GetLength(2);
        int width = targetSize.GetLength(3);

        float[,,,] res = new float[batchSize, channels, height, width];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int index = c * height * width + h * width + w;
                        res[b, c, h, w] = source[b, index];
                    }
                }
            }
        }

        return res;
    }

    #endregion

}
