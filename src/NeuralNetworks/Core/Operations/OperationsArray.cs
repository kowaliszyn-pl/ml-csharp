// Neural Networks in C♯
// File name: OperationsArray.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

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
public class OperationsArray : IOperations
{
    #region Backend Management

    public virtual OperationBackendType BackendType => OperationBackendType.CpuArrays;

    #endregion

    #region Loss Functions

    public virtual float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        // Clip the probabilities to avoid log(0) and log(1).
        float[,] clippedSoftmax = predicted.Clip(eps, 1 - eps);
        return -clippedSoftmax.Log().MultiplyElementwise(target).Mean();
    }

    public virtual float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        int batchSize = predicted.GetLength(0);
        return predicted.Subtract(target).Divide(batchSize);
    }

    public virtual float BinaryCrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1E-07F)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        // Clip the predicted probabilities to avoid log(0) and log(1).
        float[,] clippedPredicted = predicted.Clip(eps, 1 - eps);

        float[,] oneMinusPredicted = clippedPredicted.AsOnes().Subtract(clippedPredicted);
        float[,] oneMinusPredictedLog = oneMinusPredicted.Log();
        float[,] oneMinusTarget = target.AsOnes().Subtract(target);
        float[,] predictedLog = clippedPredicted.Log();

        float[,] res = target
            .MultiplyElementwise(predictedLog)
            .Add(oneMinusTarget
                .MultiplyElementwise(oneMinusPredictedLog)
            );

        return -res.Mean();
    }

    public virtual float[,] BinaryCrossEntropyLossGradient(float[,] predicted, float[,] target)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        int batchSize = predicted.GetLength(0);
        return predicted.Subtract(target).Divide(batchSize);
    }

    #endregion

    #region Activations Functions

    public virtual float[,] BipolarSigmoidOutput(float[,] input, float scale)
        => input.Sigmoid().Add(-0.5f).Multiply(scale);

    public virtual float[,] BipolarSigmoidInputGradient(float[,] outputGradient, float[,] output, float scale)
    {
        Debug.Assert(scale != 0f, "Scale must be non-zero.");

        // Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // Output = scale * (σ(x) - 0.5)  =>  σ(x) = (Output/scale) + 0.5
        // d/dx[scale * (σ(x) - 0.5)] = scale * σ(x) * (1 - σ(x))
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * scale * σ(x) * (1 - σ(x)).
        float[,] sigma = output.Divide(scale).Add(0.5f);
        float[,] sigmoidBackward = sigma.MultiplyElementwise(sigma.AsOnes().Subtract(sigma)).Multiply(scale);
        return outputGradient.MultiplyElementwise(sigmoidBackward);
    }

    public virtual float[,,,] LeakyReLUOutput(float[,,,] input, float alpha = 0.01F, float beta = 1)
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

    public virtual float[,] LeakyReLUOutput(float[,] input, float alpha = 0.01f, float beta = 1f)
    {
        int rows = input.GetLength(0);
        int columns = input.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float value = input[i, j];
                res[i, j] = value >= 0 ? value * beta : value * alpha;
            }
        }
        return res;
    }

    public virtual float[,] LeakyReLUInputGradient(float[,] outputGradient, float[,] input, float alfa, float beta)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);
        float[,] inputGradient = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                inputGradient[i, j] = input[i, j] > 0 ? outputGradient[i, j] * beta : outputGradient[i, j] * alfa;
            }
        }
        return inputGradient;
    }

    public virtual float[,] ReLUOutput(float[,] input, float beta = 1f)
    {
        int rows = input.GetLength(0);
        int columns = input.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float value = input[i, j];
                res[i, j] = value >= 0 ? value * beta : 0;
            }
        }
        return res;
    }

    public virtual float[,] ReLUInputGradient(float[,] outputGradient, float[,] input, float beta)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);
        float[,] inputGradient = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                inputGradient[i, j] = input[i, j] > 0 ? outputGradient[i, j] * beta : 0f;
            }
        }
        return inputGradient;
    }

    public virtual float[,,] ReLUOutput(float[,,] input, float beta = 1f)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);
        int dim3 = input.GetLength(2);

        float[,,] output = new float[dim1, dim2, dim3];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    float value = input[i, j, k];
                    output[i, j, k] = value >= 0 ? value * beta : 0;
                }
            }
        }
        return output;
    }

    public virtual float[,,] ReLUInputGradient(float[,,] outputGradient, float[,,] input, float beta)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);
        int dim3 = input.GetLength(2);

        float[,,] inputGradient = new float[dim1, dim2, dim3];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                        inputGradient[i, j, k] = input[i, j, k] > 0 ? outputGradient[i, j, k] * beta : 0f;
                }
            }
        }
        return inputGradient;
    }

    public virtual float[,,,] ReLUOutput(float[,,,] input, float beta = 1f)
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
                        float value = input[i, j, k, l];
                        output[i, j, k, l] = value >= 0 ? value * beta : 0;
                    }
                }
            }
        }
        return output;
    }

    public virtual float[,,,] ReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float beta)
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
                        inputGradient[i, j, k, l] = input[i, j, k, l] > 0 ? outputGradient[i, j, k, l] * beta : 0f;
                    }
                }
            }
        }
        return inputGradient;
    }

    public virtual float[,] SigmoidOutput(float[,] input)
        => input.Sigmoid();

    public virtual float[,] SigmoidInputGradient(float[,] outputGradient, float[,] output)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Sigmoid function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Sigmoid function σ(x) = 1 / (1 + exp(-x)) is σ(x) * (1 - σ(x)).
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * σ(x) * (1 - σ(x)).
        // The elementwise multiplication of the output gradient and the derivative of the Sigmoid function is returned as the input gradient.
        // σ(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient
        float[,] sigmoidBackward = output.MultiplyElementwise(output.AsOnes().Subtract(output));
        return outputGradient.MultiplyElementwise(sigmoidBackward);
    }

    public virtual float[,] SoftplusOutput(float[,] input)
        => input.Softplus();

    public virtual float[,] SoftplusInputGradient(float[,] outputGradient, float[,] output)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Softplus function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Softplus function softplus(x) = ln(1 + exp(x)) is σ(x) = 1 / (1 + exp(-x)), which is the Sigmoid function.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * σ(x).
        // The elementwise multiplication of the output gradient and the derivative of the Softplus function is returned as the input gradient.
        // σ(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient
        float[,] sigmoidBackward = output.Sigmoid();
        return outputGradient.MultiplyElementwise(sigmoidBackward);
    }

    public virtual float[,] TanhOutput(float[,] input)
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

    public virtual float[,,,] TanhOutput(float[,,,] input)
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

    #region Bias Addition Operations

    public virtual float[,] BiasAddOutput(float[,] input, float[] bias)
        => input.AddRow(bias);

    public virtual float[] BiasAddParamGradient(float[,] outputGradient)
        => outputGradient.SumByColumns();

    #endregion

    #region Bias Addition Conv1D Operations

    /// <summary>
    /// Applies a bias vector to the output of a 1-dimensional convolution operation.
    /// </summary>
    /// <remarks>
    /// Each value in the bias array is added to the corresponding channel across all batches and
    /// positions. The method does not modify the input array.
    /// </remarks>
    /// <param name="input">
    /// A three-dimensional array representing the output of a 1D convolution, with shape [batch, kernels (output channels), outputLength].
    /// </param>
    /// <param name="bias">A one-dimensional array containing the bias values to add to each channel. The length must match the number of
    /// channels in the input.</param>
    /// <returns>A three-dimensional array of the same shape as the input, with the bias added to each channel.</returns>
    public virtual float[,,] BiasAddConv1DOutput(float[,,] input, float[] bias)
    {
        int channels = input.GetLength(1);
        Debug.Assert(channels == bias.Length, "The length of the bias array must match the number of channels in the input.");

        int batchSize = input.GetLength(0);

        int outputLength = input.GetLength(2);

        float[,,] output = new float[batchSize, channels, outputLength];
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int l = 0; l < outputLength; l++)
                {
                    output[b, c, l] = input[b, c, l] + bias[c];
                }
            }
        }
        return output;
    }

    public virtual float[] BiasAddConv1DParamGradient(float[,,] outputGradient)
    {
        int channels = outputGradient.GetLength(1);
        int batchSize = outputGradient.GetLength(0);
        int outputLength = outputGradient.GetLength(2);
        float[] paramGradient = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;
            for (int b = 0; b < batchSize; b++)
            {
                for (int l = 0; l < outputLength; l++)
                {
                    sum += outputGradient[b, c, l];
                }
            }
            paramGradient[c] = sum;
        }
        return paramGradient;
    }

    #endregion

    #region Bias Addition Conv2D Operations

    public virtual float[,,,] BiasAddConv2DOutput(float[,,,] input, float[] bias)
    {
        // Input: [batch, kernels, outputHeight, outputWidth]
        int batchSize = input.GetLength(0);
        int channels = input.GetLength(1);
        int outputHeight = input.GetLength(2);
        int outputWidth = input.GetLength(3);

        Debug.Assert(channels == bias.Length, "The length of the bias array must match the number of channels in the input.");

        float[,,,] output = new float[batchSize, channels, outputHeight, outputWidth];
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outputHeight; h++)
                {
                    for (int w = 0; w < outputWidth; w++)
                    {
                        output[b, c, h, w] = input[b, c, h, w] + bias[c];
                    }
                }
            }
        }

        return output;
    }

    public virtual float[] BiasAddConv2DParamGradient(float[,,,] outputGradient)
    {
        int channels = outputGradient.GetLength(1);
        int batchSize = outputGradient.GetLength(0);
        int outputHeight = outputGradient.GetLength(2);
        int outputWidth = outputGradient.GetLength(3);
        float[] paramGradient = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;
            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < outputHeight; h++)
                {
                    for (int w = 0; w < outputWidth; w++)
                    {
                        sum += outputGradient[b, c, h, w];
                    }
                }
            }
            paramGradient[c] = sum;
        }
        return paramGradient;
    }

    #endregion

    #region Convolution 1D Operations

    public virtual float[,,] Convolve1DOutput(float[,,] input, float[,,] weights, int padding, int stride = 1, int dilatation = 1)
    {
        int batchSize = input.GetLength(0);
        int inputChannels = input.GetLength(1);
        int inputLength = input.GetLength(2);

        int outputChannels = weights.GetLength(1);
        int kernelLength = weights.GetLength(2);

        Debug.Assert(weights.GetLength(0) == inputChannels);

        int effectiveInputLength = inputLength + 2 * padding;
        int effectiveKernelLength = kernelLength + (dilatation - 1) * (kernelLength - 1);

        int outputLength = (effectiveInputLength - effectiveKernelLength) / stride + 1;

        float[,,] output = new float[batchSize, outputChannels, outputLength];

        for (int b = 0; b < batchSize; b++)
        {
            for (int oc = 0; oc < outputChannels; oc++)
            {
                for (int ol = 0; ol < outputLength; ol++)
                {
                    float sum = 0.0f;
                    for (int ic = 0; ic < inputChannels; ic++)
                    {
                        for (int kl = 0; kl < kernelLength; kl++)
                        {
                            int il = ol * stride + kl * dilatation - padding;
                            if (il >= 0 && il < inputLength)
                            {
                                sum += input[b, ic, il] * weights[ic, oc, kl];
                            }
                        }
                    }
                    output[b, oc, ol] = sum;
                }
            }
        }

        return output;
    }

    public virtual float[,,] Convolve1DInputGradient(float[,,] input, float[,,] weights, float[,,] outputGradient, int padding, int stride = 1, int dilatation = 0)
    {
        int inputChannels = input.GetLength(1);
        int inputLength = input.GetLength(2);

        int batchSize = outputGradient.GetLength(0);
        int outputChannels = outputGradient.GetLength(1);
        int outputLength = outputGradient.GetLength(2);

        int kernelLength = weights.GetLength(2);

        Debug.Assert(weights.GetLength(0) == inputChannels);

        float[,,] inputGradient = new float[batchSize, inputChannels, inputLength];

        for (int b = 0; b < batchSize; b++)
        {
            for (int ic = 0; ic < inputChannels; ic++)
            {
                for (int il = 0; il < inputLength; il++)
                {
                    float sum = 0.0f;
                    for (int oc = 0; oc < outputChannels; oc++)
                    {
                        for (int kl = 0; kl < kernelLength; kl++)
                        {
                            int ol = (il + padding - kl * dilatation) / stride;
                            if (ol >= 0 && ol < outputLength && (il + padding - kl * dilatation) % stride == 0)
                            {
                                sum += outputGradient[b, oc, ol] * weights[ic, oc, kl];
                            }
                        }
                    }
                    inputGradient[b, ic, il] += sum;
                }
            }
        }
        return inputGradient;
    }

    public virtual float[,,] Convolve1DParamGradient(float[,,] input, float[,,] outputGradient, int kernelLength, int padding, int stride, int dilatation)
    {
        int inputChannels = input.GetLength(1);
        int inputLength = input.GetLength(2);

        int batchSize = outputGradient.GetLength(0);
        int outputChannels = outputGradient.GetLength(1);
        int outputLength = outputGradient.GetLength(2);

        float[,,] paramGradient = new float[inputChannels, outputChannels, kernelLength];
        for (int b = 0; b < batchSize; b++)
        {
            for (int ic = 0; ic < inputChannels; ic++)
            {
                for (int oc = 0; oc < outputChannels; oc++)
                {
                    for (int kl = 0; kl < kernelLength; kl++)
                    {
                        float sum = 0.0f;
                        for (int ol = 0; ol < outputLength; ol++)
                        {
                            int il = ol * stride + kl * dilatation - padding;
                            if (il >= 0 && il < inputLength)
                            {
                                sum += outputGradient[b, oc, ol] * input[b, ic, il];
                            }
                        }
                        paramGradient[ic, oc, kl] += sum;
                    }
                }
            }
        }
        return paramGradient;
    }

    #endregion

    #region Convolution 2D Operations

    public virtual float[,,,] Convolve2DOutput(float[,,,] input, float[,,,] weights, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1)
    {
        int batchSize = input.GetLength(0);
        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int weightChannels = weights.GetLength(0);
        int outputChannels = weights.GetLength(1);
        int kernelHeight = weights.GetLength(2);
        int kernelWidth = weights.GetLength(3);

        Debug.Assert(weightChannels == inputChannels);

        int effectiveInputHeight = inputHeight + 2 * paddingHeight;
        int effectiveInputWidth = inputWidth + 2 * paddingWidth;

        int effectiveKernelHeight = kernelHeight + (dilatationHeight - 1) * (kernelHeight - 1);
        int effectiveKernelWidth = kernelWidth + (dilatationWidth - 1) * (kernelWidth - 1);

        int outputHeight = (effectiveInputHeight - effectiveKernelHeight) / strideHeight + 1;
        int outputWidth = (effectiveInputWidth - effectiveKernelWidth) / strideWidth + 1;

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
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                int ih = oh * strideHeight + kh * dilatationHeight - paddingHeight;
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int iw = ow * strideWidth + kw * dilatationWidth - paddingWidth;
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

    public virtual float[,,,] Convolve2DInputGradient(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1)
    {
        int batchSize = outputGradient.GetLength(0);
        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        int kernelHeight = weights.GetLength(2);
        int kernelWidth = weights.GetLength(3);

        Debug.Assert(weights.GetLength(0) == inputChannels);

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
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int oh = Math.DivRem(ih + paddingHeight - kh * dilatationHeight, strideHeight, out int remH);
                                    int ow = Math.DivRem(iw + paddingWidth - kw * dilatationWidth, strideWidth, out int remW);
                                    if (oh >= 0 && oh < outputGradientHeight && remH == 0
                                        && ow >= 0 && ow < outputGradientWidth && remW == 0
                                    )
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

    public virtual float[,,,] Convolve2DParamGradient(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1)
    {
        int batchSize = outputGradient.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

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
                                    int ih = oh * strideHeight + kh * dilatationHeight - paddingHeight;
                                    int iw = ow * strideWidth + kw * dilatationWidth - paddingWidth;
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

    #endregion

    #region Weight Multiplication Operations

    public virtual float[,] WeightMultiplyOutput(float[,] input, float[,] weights)
    => input.MultiplyDot(weights);

    public virtual float[,] WeightMultiplyInputGradient(float[,] outputGradient, float[,] weights)
        => outputGradient.MultiplyDot(weights.Transpose());

    public virtual float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient)
        => input.Transpose().MultiplyDot(outputGradient);

    #endregion

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

    public virtual float[,] DropoutInputGradient(float[,] outputGradient, float[,] mask)
        => outputGradient.MultiplyElementwise(mask);

    public virtual float[,,] DropoutOutput(float[,,] input, bool inference, float keepProb, SeededRandom? random, out float[,,]? mask)
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

    public virtual float[,,] DropoutInputGradient(float[,,] outputGradient, float[,,] mask)
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

    public virtual float[,,] MaxPooling1DOutput(float[,,] input, int size, out int[,,] maxIndices)
    {
        int batchSize = input.GetLength(0);
        int channels = input.GetLength(1);
        int length = input.GetLength(2);

        int outputLength = length / size;

        float[,,] output = new float[batchSize, channels, outputLength];
        maxIndices = new int[batchSize, channels, outputLength];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int ol = 0; ol < outputLength; ol++)
                {
                    float maxVal = float.NegativeInfinity;
                    int maxIdx = -1;
                    for (int i = 0; i < size; i++)
                    {
                        int idx = ol * size + i;
                        if (idx < length)
                        {
                            float val = input[b, c, idx];
                            if (val > maxVal)
                            {
                                maxVal = val;
                                maxIdx = idx;
                            }
                        }
                    }
                    output[b, c, ol] = maxVal;
                    maxIndices[b, c, ol] = maxIdx;
                }
            }
        }
        return output;
    }

    public virtual float[,,] MaxPooling1DInputGradient(float[,,] input, float[,,] outputGradient, int size, int[,,] maxIndices)
    {
        int inputLength = input.GetLength(2);
        int batchSize = outputGradient.GetLength(0);
        int channels = outputGradient.GetLength(1);

        int outputLength = outputGradient.GetLength(2);
        float[,,] inputGradient = new float[batchSize, channels, inputLength];
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int ol = 0; ol < outputLength; ol++)
                {
                    int maxIdx = maxIndices[b, c, ol];
                    if (maxIdx >= 0 && maxIdx < inputLength)
                    {
                        inputGradient[b, c, maxIdx] += outputGradient[b, c, ol];
                    }
                }
            }
        }
        return inputGradient;
    }

    public virtual float[,] GlobalAveragePooling1DOutput(float[,,] input)
    {
        int batchSize = input.GetLength(0);
        int channels = input.GetLength(1);
        int length = input.GetLength(2);

        float[,] output = new float[batchSize, channels];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float sum = 0f;
                for (int l = 0; l < length; l++)
                {
                    sum += input[b, c, l];
                }
                output[b, c] = sum / length;
            }
        }
        return output;
    }

    public virtual float[,,] GlobalAveragePooling1DInputGradient(float[,,] input, float[,] outputGradient)
    {
        int inputLength = input.GetLength(2);

        int channels = input.GetLength(1);
        int length = input.GetLength(2);
        int batchSize = outputGradient.GetLength(0);

        float[,,] inputGradient = new float[batchSize, channels, length];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float distributedGrad = outputGradient[b, c] / inputLength;
                for (int l = 0; l < length; l++)
                {
                    inputGradient[b, c, l] = distributedGrad;
                }
            }
        }

        return inputGradient;
    }

    #endregion

}
