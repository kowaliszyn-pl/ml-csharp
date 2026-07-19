// Neural Networks in C♯
// File name: OperationsSpanParallel.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NeuralNetworks.Core.Operations;

public class OperationsSpanParallel : OperationsSpan
{
    #region Backend Management

    public override OperationBackendType BackendType => OperationBackendType.CpuSpansParallel;

    #endregion

    #region Loss Functions

    public override float[,] SoftmaxCrossEntropyLossGradient(float[,] softmaxOutput, float[,] target)
    {
        Debug.Assert(softmaxOutput.Length == target.Length, "Predicted and target arrays must have the same length.");

        int batchSize = softmaxOutput.GetLength(0);

        Debug.Assert(batchSize > 0, "Batch size must be greater than zero.");

        int numClasses = softmaxOutput.GetLength(1);
        float[,] gradient = new float[batchSize, numClasses];

        Parallel.For(0, gradient.Length, i =>
        {
            ReadOnlySpan<float> predictedSpan = MemoryMarshal.CreateReadOnlySpan(ref softmaxOutput[0, 0], softmaxOutput.Length);
            ReadOnlySpan<float> targetSpan = MemoryMarshal.CreateReadOnlySpan(ref target[0, 0], target.Length);
            Span<float> gradientSpan = MemoryMarshal.CreateSpan(ref gradient[0, 0], gradient.Length);

            gradientSpan[i] = (predictedSpan[i] - targetSpan[i]) / batchSize;
        });

        return gradient;
    }

    public override float[,,,] MeanSquaredErrorLossGradient(float[,,,] errors, MseReduction mseReduction)
    {
        int batchSize = errors.GetLength(0);

        Debug.Assert(batchSize > 0, "Batch size must be greater than zero.");

        float[,,,] gradient = new float[batchSize, errors.GetLength(1), errors.GetLength(2), errors.GetLength(3)];

        float scaleFactor = (mseReduction == MseReduction.ElementMean) ? 2f / errors.Length : 2f / batchSize;
        int gradientSpanLength = gradient.Length;
        //Parallel.For, gradient.Length, i =>
        //{
        //    ReadOnlySpan<float> errorsSpan = MemoryMarshal.CreateReadOnlySpan(ref errors[0, 0, 0, 0], errors.Length);
        //    Span<float> gradientSpan = MemoryMarshal.CreateSpan(ref gradient[0, 0, 0, 0], gradient.Length);

        //    gradientSpan[i] = errorsSpan[i] * scaleFactor;
        //});

        Parallel.ForEach(Partitioner.Create(0, gradientSpanLength), range =>
        {
            ReadOnlySpan<float> errorsSpan = MemoryMarshal.CreateReadOnlySpan(ref errors[0, 0, 0, 0], errors.Length);
            Span<float> gradientSpan = MemoryMarshal.CreateSpan(ref gradient[0, 0, 0, 0], gradientSpanLength);

            for (int i = range.Item1; i < range.Item2; i++)
            {
                gradientSpan[i] = errorsSpan[i] * scaleFactor;
            }
        });

        return gradient;
    }

    #endregion

    #region Activations Functions

    public override float[,,,] LeakyReLUOutput(float[,,,] input, float alpha = 0.01f, float beta = 1f)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);
        int dim3 = input.GetLength(2);
        int dim4 = input.GetLength(3);

        float[,,,] output = new float[dim1, dim2, dim3, dim4];

        Parallel.For(0, input.Length, i =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

            float value = inputSpan[i];
            outputSpan[i] = value >= 0 ? value * beta : value * alpha;
        });

        return output;
    }

    public override float[,,,] LeakyReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta)
    {
        int dim1 = outputGradient.GetLength(0);
        int dim2 = outputGradient.GetLength(1);
        int dim3 = outputGradient.GetLength(2);
        int dim4 = outputGradient.GetLength(3);

        Debug.Assert(dim1 > 0 && dim2 > 0 && dim3 > 0 && dim4 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(input.GetLength(0) == dim1 && input.GetLength(1) == dim2 && input.GetLength(2) == dim3 && input.GetLength(3) == dim4, "Shapes of outputGradient and input must match for elementwise operations.");

        float[,,,] inputGradient = new float[dim1, dim2, dim3, dim4];

        Parallel.For(0, input.Length, i =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);

            inputGradientSpan[i] = inputSpan[i] > 0 ? outputGradientSpan[i] * beta : outputGradientSpan[i] * alfa;
        });

        return inputGradient;
    }

    public override float[,,,] ReLUOutput(float[,,,] input, float beta = 1)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);
        int dim3 = input.GetLength(2);
        int dim4 = input.GetLength(3);
        float[,,,] output = new float[dim1, dim2, dim3, dim4];

        Parallel.For(0, input.Length, i =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

            float value = inputSpan[i];
            outputSpan[i] = value >= 0 ? value * beta : 0f;
        });
        return output;
    }

    public override float[,,,] ReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float beta)
    {
        int dim1 = outputGradient.GetLength(0);
        int dim2 = outputGradient.GetLength(1);
        int dim3 = outputGradient.GetLength(2);
        int dim4 = outputGradient.GetLength(3);

        Debug.Assert(dim1 > 0 && dim2 > 0 && dim3 > 0 && dim4 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(input.GetLength(0) == dim1 && input.GetLength(1) == dim2 && input.GetLength(2) == dim3 && input.GetLength(3) == dim4, "Shapes of outputGradient and input must match for elementwise operations.");

        float[,,,] inputGradient = new float[dim1, dim2, dim3, dim4];

        Parallel.For(0, input.Length, i =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);
            inputGradientSpan[i] = inputSpan[i] > 0 ? outputGradientSpan[i] * beta : 0f;
        });

        return inputGradient;
    }

    public override float[,] SoftsignOutput(float[,] input)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);

        float[,] output = new float[dim1, dim2];

        Parallel.For(0, output.Length, i =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0], input.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0], output.Length);

            outputSpan[i] = inputSpan[i] / (MathF.Abs(inputSpan[i]) + 1f);
        });

        return output;
    }

    public override float[,] SoftsignInputGradient(float[,] outputGradient, float[,] input)
    {
        int dim0 = outputGradient.GetLength(0);
        int dim1 = outputGradient.GetLength(1);

        Debug.Assert(input.GetLength(0) == dim0 && input.GetLength(1) == dim1, "Shapes of outputGradient and input must match for elementwise operations.");

        float[,] inputGradient = new float[dim0, dim1];
        Parallel.For(0, inputGradient.Length, i =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0], input.Length);
            Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0], inputGradient.Length);

            float onePlusAbs = MathF.Abs(inputSpan[i]) + 1f;
            inputGradientSpan[i] = outputGradientSpan[i] / (onePlusAbs * onePlusAbs);
        });

        return inputGradient;
    }

    public override float[,,,] TanhOutput(float[,,,] input)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);
        int dim3 = input.GetLength(2);
        int dim4 = input.GetLength(3);

        float[,,,] output = new float[dim1, dim2, dim3, dim4];

        Parallel.For(0, output.Length, i =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

            outputSpan[i] = MathF.Tanh(inputSpan[i]);
        });

        return output;
    }

    public override float[,,,] TanhInputGradient(float[,,,] outputGradient, float[,,,] output)
    {
        int dim0 = outputGradient.GetLength(0);
        int dim1 = outputGradient.GetLength(1);
        int dim2 = outputGradient.GetLength(2);
        int dim3 = outputGradient.GetLength(3);

        Debug.Assert(output.GetLength(0) == dim0 && output.GetLength(1) == dim1 && output.GetLength(2) == dim2 && output.GetLength(3) == dim3, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,,,] inputGradient = new float[dim0, dim1, dim2, dim3];

        Parallel.For(0, inputGradient.Length, i =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
            ReadOnlySpan<float> outputSpan = MemoryMarshal.CreateReadOnlySpan(ref output[0, 0, 0, 0], output.Length);
            Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);

            float y = outputSpan[i];
            inputGradientSpan[i] = outputGradientSpan[i] * (1f - (y * y));
        });

        return inputGradient;
    }

    public override float[,] TanhOutput(float[,] input)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);

        float[,] output = new float[dim1, dim2];

        Parallel.For(0, output.Length, i =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0], input.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0], output.Length);

            outputSpan[i] = MathF.Tanh(inputSpan[i]);
        });

        return output;
    }

    public override float[,] TanhInputGradient(float[,] outputGradient, float[,] output)
    {
        int dim0 = outputGradient.GetLength(0);
        int dim1 = outputGradient.GetLength(1);

        Debug.Assert(output.GetLength(0) == dim0 && output.GetLength(1) == dim1, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,] inputGradient = new float[dim0, dim1];

        Parallel.For(0, inputGradient.Length, i =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
            ReadOnlySpan<float> outputSpan = MemoryMarshal.CreateReadOnlySpan(ref output[0, 0], output.Length);
            Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0], inputGradient.Length);

            float y = outputSpan[i];
            inputGradientSpan[i] = outputGradientSpan[i] * (1f - (y * y));
        });

        return inputGradient;
    }

    public override float[,] TanhInputScaledOutput(float[,] input, float scale)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);

        float[,] output = new float[dim1, dim2];
        float reciprocalScale = 1f / scale;

        Parallel.For(0, output.Length, i =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0], input.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0], output.Length);

            outputSpan[i] = MathF.Tanh(inputSpan[i] * reciprocalScale);
        });

        return output;
    }

    public override float[,] TanhInputScaledInputGradient(float[,] outputGradient, float[,] output, float scale)
    {
        int dim0 = outputGradient.GetLength(0);
        int dim1 = outputGradient.GetLength(1);

        Debug.Assert(output.GetLength(0) == dim0 && output.GetLength(1) == dim1, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,] inputGradient = new float[dim0, dim1];
        float reciprocalScale = 1f / scale;

        Parallel.For(0, inputGradient.Length, i =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
            ReadOnlySpan<float> outputSpan = MemoryMarshal.CreateReadOnlySpan(ref output[0, 0], output.Length);
            Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0], inputGradient.Length);

            float y = outputSpan[i];
            inputGradientSpan[i] = outputGradientSpan[i] * (1f - (y * y)) * reciprocalScale;
        });

        return inputGradient;
    }

    #endregion

    #region Parametric Operations

    #region Bias Addition Operations

    public override float[,] BiasAddOutput(float[,] input, float[] bias)
    {
        int batchSize = input.GetLength(0);
        int features = input.GetLength(1);

        Debug.Assert(bias.Length == features, "Bias length must match the number of features in the input.");

        float[,] output = new float[batchSize, features];
        Parallel.For(0, output.Length, i =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0], input.Length);
            ReadOnlySpan<float> biasSpan = bias.AsSpan();
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0], output.Length);

            int featureIndex = i % features;
            outputSpan[i] = inputSpan[i] + biasSpan[featureIndex];
        });
        return output;
    }

    public override float[] BiasAddParamGradient(float[,] outputGradient)
    {
        int batchSize = outputGradient.GetLength(0);
        int features = outputGradient.GetLength(1);
        float[] paramGradient = new float[features];
        Parallel.For(0, features, feature =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
            Span<float> paramGradientSpan = paramGradient.AsSpan();
            float sum = 0f;
            for (int i = 0; i < batchSize; i++)
            {
                sum += outputGradientSpan[i * features + feature];
            }
            paramGradientSpan[feature] = sum;
        });
        return paramGradient;
    }

    #endregion

    #region Bias Addition Conv2D Operations

    public override float[,,,] BiasAddConv2DOutput(float[,,,] input, float[] bias)
    {
        int batchSize = input.GetLength(0);
        int channels = input.GetLength(1);
        int height = input.GetLength(2);
        int width = input.GetLength(3);
        int spatialSize = height * width;

        Debug.Assert(bias.Length == channels, "Bias length must match the number of channels in the input.");

        float[,,,] output = new float[batchSize, channels, height, width];
        Parallel.For(0, output.Length, i =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            ReadOnlySpan<float> biasSpan = MemoryMarshal.CreateReadOnlySpan(ref bias[0], bias.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

            int channelIndex = (i / spatialSize) % channels;
            outputSpan[i] = inputSpan[i] + biasSpan[channelIndex];
        });

        return output;
    }

    public override float[] BiasAddConv2DParamGradient(float[,,,] outputGradient)
    {
        int batchSize = outputGradient.GetLength(0);
        int channels = outputGradient.GetLength(1);
        int height = outputGradient.GetLength(2);
        int width = outputGradient.GetLength(3);
        int itemsPerChannel = height * width;
        int elementCount = batchSize * itemsPerChannel;

        float[] paramGradient = new float[channels];

        Parallel.For(0, channels, channel =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
            Span<float> paramGradientSpan = paramGradient.AsSpan();

            int channelOffset = channel * itemsPerChannel;
            float sum = 0f;

            for (int i = 0; i < elementCount; i++)
            {
                sum += outputGradientSpan[channelOffset + i];
            }
            paramGradient[channel] = sum;
        });

        return paramGradient;
    }

    #endregion

    #region Convolution 2D Operations

    public override float[,,,] Convolve2DOutput(float[,,,] input, float[,,,] weights, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1)
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

        // pre-compute sizes for offsets
        int outputCSize = outputHeight * outputWidth;
        int outputBSize = outputChannels * outputCSize;
        int inputCSize = inputHeight * inputWidth;
        int inputBSize = inputChannels * inputCSize;
        int weightsOutputCSize = kernelHeight * kernelWidth;
        int weightsCSize = outputChannels * weightsOutputCSize;

        Parallel.ForEach(Partitioner.Create(0, batchSize), range =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0, 0, 0], weights.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

            for (int b = range.Item1; b < range.Item2; b++)
            {
                int inputBIndex = b * inputBSize;
                int outputBIndex = b * outputBSize;

                //Parallel.For(0, outputChannels, oc =>
                for (int oc = 0; oc < outputChannels; oc++)
                {
                    int weightsOutputCIndex = oc * weightsOutputCSize;
                    int outputCIndex = oc * outputCSize;
                    for (int oh = 0; oh < outputHeight; oh++) // ~28
                    {
                        int outputHIndex = oh * outputWidth;
                        int ohMinusPad = oh * strideHeight - paddingHeight;
                        for (int ow = 0; ow < outputWidth; ow++) // ~28
                        {
                            int owMinusPad = ow * strideWidth - paddingWidth;
                            float sum = 0f;
                            for (int ic = 0; ic < inputChannels; ic++) // 1 (black&white) or 3 (RGB)
                            {
                                int inputCIndex = ic * inputCSize;
                                int weightsInputCIndex = ic * weightsCSize;
                                for (int kh = 0; kh < kernelHeight; kh++) // ~3
                                {
                                    int weightsKernelHIndex = kh * kernelWidth;
                                    // int ih = oh * strideHeight + kh * dilatationHeight - paddingHeight;
                                    int ih = kh * dilatationHeight + ohMinusPad;
                                    if (ih >= 0 && ih < inputHeight)
                                    {
                                        int inputHIndex = ih * inputWidth;
                                        for (int kw = 0; kw < kernelWidth; kw++) // ~3
                                        {
                                            // int iw = ow * strideWidth + kw * dilatationWidth - paddingWidth;
                                            int iw = kw * dilatationWidth + owMinusPad;
                                            if (iw >= 0 && iw < inputWidth)
                                            {
                                                // sum += input[b, ic, ih, iw] * weights[ic, oc, kh, kw];
                                                sum += inputSpan[inputBIndex + inputCIndex + inputHIndex + iw] *
                                                       weightsSpan[weightsInputCIndex + weightsOutputCIndex + weightsKernelHIndex + kw];
                                            }
                                        }
                                    }
                                }
                            }
                            // output[b, oc, oh, ow] = sum;
                            outputSpan[outputBIndex + outputCIndex + outputHIndex + ow] = sum;
                        }
                    }
                }
            }
        });

        return output;
    }

    public override float[,,,] Convolve2DInputGradient(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1)
    {
        int batchSize = outputGradient.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputGradientChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        int kernelHeight = weights.GetLength(2);
        int kernelWidth = weights.GetLength(3);

        Debug.Assert(weights.GetLength(0) == inputChannels);
        Debug.Assert(weights.GetLength(1) == outputGradientChannels);

        float[,,,] inputGradient = new float[batchSize, inputChannels, inputHeight, inputWidth];

        // pre-compute sizes for offsets
        int outputGradientCSize = outputGradientHeight * outputGradientWidth;
        int outputGradientBSize = outputGradientChannels * outputGradientCSize;
        int weightsOutputCSize = kernelHeight * kernelWidth;
        int weightsInputCSize = outputGradientChannels * weightsOutputCSize;
        int inputGradientCSize = inputHeight * inputWidth;
        int inputGradientBSize = inputChannels * inputGradientCSize;

        Parallel.ForEach(Partitioner.Create(0, batchSize), range =>
        {
            ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0, 0, 0], weights.Length);
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
            Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);

            for (int b = range.Item1; b < range.Item2; b++)
            {
                int outputGradientBIndex = b * outputGradientBSize;
                int inputGradientBIndex = b * inputGradientBSize;

                //Parallel.For(0, inputChannels, ic =>
                for (int ic = 0; ic < inputChannels; ic++)
                {
                    {
                        int weightsInputCIndex = ic * weightsInputCSize;
                        int inputGradientCIndex = ic * inputGradientCSize;
                        for (int ih = 0; ih < inputHeight; ih++)
                        {
                            int inputGradientHIndex = ih * inputWidth;
                            int ihPlusPad = ih + paddingHeight;
                            for (int iw = 0; iw < inputWidth; iw++)
                            {
                                float sum = 0f;

                                int iwPlusPad = iw + paddingWidth;
                                for (int oc = 0; oc < outputGradientChannels; oc++)
                                {
                                    int outputGradientCIndex = oc * outputGradientCSize;
                                    int weightsOutputCIndex = oc * weightsOutputCSize;
                                    for (int kh = 0; kh < kernelHeight; kh++)
                                    {
                                        int oh = Math.DivRem(ihPlusPad - kh * dilatationHeight, strideHeight, out int remH);
                                        // int oh = ihPlusPad - kh;
                                        if (oh >= 0 && oh < outputGradientHeight && remH == 0)
                                        {
                                            int weightsKernelHIndex = kh * kernelWidth;
                                            int outputHIndex = oh * outputGradientWidth;
                                            for (int kw = 0; kw < kernelWidth; kw++)
                                            {
                                                int ow = Math.DivRem(iwPlusPad - kw * dilatationWidth, strideWidth, out int remW);
                                                // int ow = iwPlusPad - kw;
                                                if (ow >= 0 && ow < outputGradientWidth && remW == 0)
                                                {
                                                    // sum += outputGradient[b, oc, oh, ow] * weights[ic, oc, kh, kw];
                                                    sum += outputGradientSpan[outputGradientBIndex + outputGradientCIndex + outputHIndex + ow]
                                                        * weightsSpan[weightsInputCIndex + weightsOutputCIndex + weightsKernelHIndex + kw];
                                                }
                                            }
                                        }
                                    }
                                }
                                // inputGradient[b, ic, ih, iw] = sum;
                                inputGradientSpan[inputGradientBIndex + inputGradientCIndex + inputGradientHIndex + iw] = sum;
                            }
                        }
                    }
                }
            }
        });

        return inputGradient;
    }

    public override float[,,,] Convolve2DParamGradient(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1)
    {
        int batchSize = outputGradient.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputGradientChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        float[,,,] paramGradient = new float[inputChannels, outputGradientChannels, kernelHeight, kernelWidth];

        // pre-compute sizes for offsets
        int outputGradientOutputCSize = outputGradientHeight * outputGradientWidth;
        int outputGradientBSize = outputGradientChannels * outputGradientOutputCSize;
        int inputCSize = inputHeight * inputWidth;
        int inputBSize = inputChannels * inputCSize;
        int paramGradientOutputCSize = kernelHeight * kernelWidth;
        int paramGradientInputCSize = outputGradientChannels * paramGradientOutputCSize;

        // Thread-local storage for accumulating gradients per batch range
        ConcurrentBag<float[,,,]> threadLocalGradients = [];

        Parallel.ForEach(Partitioner.Create(0, batchSize), range =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);

            // Create thread-local gradient array for this batch range
            float[,,,] localParamGradient = new float[inputChannels, outputGradientChannels, kernelHeight, kernelWidth];
            Span<float> localParamGradientSpan = MemoryMarshal.CreateSpan(ref localParamGradient[0, 0, 0, 0], localParamGradient.Length);

            for (int b = range.Item1; b < range.Item2; b++)
            {
                int outputGradientBIndex = b * outputGradientBSize;
                int inputBIndex = b * inputBSize;

                // Parallel.For(0, inputChannels, ic =>
                for (int ic = 0; ic < inputChannels; ic++)
                {
                    int inputCIndex = ic * inputCSize;
                    int paramGradientInputCIndex = ic * paramGradientInputCSize;

                    for (int oc = 0; oc < outputGradientChannels; oc++)
                    {
                        int outputGradientOutputCIndex = oc * outputGradientOutputCSize;
                        int paramGradientOutputCIndex = oc * paramGradientOutputCSize;
                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            int paramGradientKernelHIndex = kh * kernelWidth;
                            int khMinusPad = kh * dilatationHeight - paddingHeight;
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int kwMinusPad = kw * dilatationWidth - paddingWidth;
                                float sum = 0f;
                                for (int oh = 0; oh < outputGradientHeight; oh++)
                                {
                                    // int ih = oh * strideHeight + kh * dilatationHeight - paddingHeight;
                                    int ih = oh * strideHeight + khMinusPad;
                                    if (ih >= 0 && ih < inputHeight)
                                    {
                                        int inputHIndex = ih * inputWidth;
                                        int outputGradientHIndex = oh * outputGradientWidth;
                                        for (int ow = 0; ow < outputGradientWidth; ow++)
                                        {
                                            // int iw = ow * strideWidth + kw * dilatationWidth - paddingWidth;
                                            int iw = ow * strideWidth + kwMinusPad;
                                            if (iw >= 0 && iw < inputWidth)
                                            {
                                                // sum += outputGradient[b, oc, oh, ow] * input[b, ic, ih, iw]
                                                sum += outputGradientSpan[outputGradientBIndex + outputGradientOutputCIndex + outputGradientHIndex + ow] *
                                                       inputSpan[inputBIndex + inputCIndex + inputHIndex + iw];
                                            }
                                        }
                                    }
                                }
                                // Accumulate in thread-local array (no race condition)
                                localParamGradientSpan[paramGradientInputCIndex + paramGradientOutputCIndex + paramGradientKernelHIndex + kw] += sum;
                            }
                        }
                    }
                }
            }

            // Store thread-local gradient for sequential summation
            threadLocalGradients.Add(localParamGradient);
        });

        // Sequential summation of all thread-local gradients into final result
        Span<float> paramGradientSpan = MemoryMarshal.CreateSpan(ref paramGradient[0, 0, 0, 0], paramGradient.Length);
        foreach (float[,,,] localGradient in threadLocalGradients)
        {
            ReadOnlySpan<float> localGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref localGradient[0, 0, 0, 0], localGradient.Length);
            for (int i = 0; i < paramGradientSpan.Length; i++)
            {
                paramGradientSpan[i] += localGradientSpan[i];
            }
        }

        return paramGradient;
    }

    #endregion

    #region Weight Multiplication Operations

    public override float[,] WeightMultiplyOutput(float[,] input, float[,] weights)
    {
        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = weights.GetLength(1);

        Debug.Assert(weights.GetLength(0) == inputFeatures, "Input features must match weight input dimension.");

        float[,] output = new float[batchSize, outputFeatures];

        Parallel.ForEach(Partitioner.Create(0, batchSize), range =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0], input.Length);
            ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0], weights.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0], output.Length);

            for (int b = range.Item1; b < range.Item2; b++)
            {
                int inputBIndex = b * inputFeatures;
                int outputBIndex = b * outputFeatures;
                for (int ofeature = 0; ofeature < outputFeatures; ofeature++)
                {
                    float sum = 0f;
                    for (int ifeature = 0; ifeature < inputFeatures; ifeature++)
                    {
                        // sum += input[b, inputFeature] * weights[inputFeature, outputFeature];
                        sum += inputSpan[inputBIndex + ifeature] * weightsSpan[ifeature * outputFeatures + ofeature];
                    }
                    // output[b, outputFeature] = sum;
                    outputSpan[outputBIndex + ofeature] = sum;
                }
            }
        });

        return output;
    }

    public override float[,] WeightMultiplyInputGradient(float[,] outputGradient, float[,] weights)
    {
        int batchSize = outputGradient.GetLength(0); // 100
        int inputFeatures = weights.GetLength(0); // 46
        int outputFeatures = weights.GetLength(1); // 10
        int outputGradientFeatures = outputGradient.GetLength(1); // 10

        Debug.Assert(outputGradientFeatures == outputFeatures, "Output features of output gradient must match weight output dimension.");

        float[,] inputGradient = new float[batchSize, inputFeatures];

        Parallel.ForEach(Partitioner.Create(0, batchSize), range =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
            ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0], weights.Length);
            Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0], inputGradient.Length);
            for (int b = range.Item1; b < range.Item2; b++)
            {
                int outputGradientBIndex = b * outputFeatures;
                int inputGradientBIndex = b * inputFeatures;

                //Parallel.For(0, inputFeatures, inputFeature =>
                for (int inputFeature = 0; inputFeature < inputFeatures; inputFeature++)
                {
                    //ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
                    //ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0], weights.Length);
                    //Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0], inputGradient.Length);
                    int inputFeatureIndex = inputFeature * outputFeatures;
                    float sum = 0f;
                    for (int outputFeature = 0; outputFeature < outputFeatures; outputFeature++)
                    {
                        // sum += outputGradient[b, outputFeature] * weights[inputFeature, outputFeature];
                        sum += outputGradientSpan[outputGradientBIndex + outputFeature] * weightsSpan[inputFeatureIndex + outputFeature];
                    }
                    // inputGradient[b, inputFeature] = sum;
                    inputGradientSpan[inputGradientBIndex + inputFeature] = sum;
                } //);
            }
        });

        return inputGradient;
    }

    public override float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient)
    {
        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = outputGradient.GetLength(1);

        Debug.Assert(outputGradient.GetLength(0) == batchSize, "Batch size of output gradient must match batch size of input.");

        float[,] paramGradient = new float[inputFeatures, outputFeatures];

        Parallel.ForEach(Partitioner.Create(0, inputFeatures), range =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0], input.Length);
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
            Span<float> paramGradientSpan = MemoryMarshal.CreateSpan(ref paramGradient[0, 0], paramGradient.Length);
            for (int ifeature = range.Item1; ifeature < range.Item2; ifeature++)
            {
                int paramGradientInputFeatureIndex = ifeature * outputFeatures;

                // Parallel.For(0, outputFeatures, ofeature =>
                for (int ofeature = 0; ofeature < outputFeatures; ofeature++)
                {
                    float sum = 0f;
                    for (int b = 0; b < batchSize; b++)
                    {
                        // sum += input[b, inputFeature] * outputGradient[b, outputFeature];
                        sum += inputSpan[b * inputFeatures + ifeature] * outputGradientSpan[b * outputFeatures + ofeature];
                    }
                    // paramGradient[inputFeature, outputFeature] = sum;
                    paramGradientSpan[paramGradientInputFeatureIndex + ofeature] = sum;
                }
            }
        });
        return paramGradient;
    }

    #endregion

    #endregion

    #region Transformations

    public override float[,,,] MaxPooling2DOutput(float[,,,] input, int sizeHeight, int sizeWidth, out (int MaxIndexH, int MaxIndexW)[,,,] maxIndices)
    {
        int batchSize = input.GetLength(0);
        int channels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        Debug.Assert(inputHeight % sizeHeight == 0, "Input height must be divisible by pooling size height.");
        Debug.Assert(inputWidth % sizeWidth == 0, "Input width must be divisible by pooling size width.");

        int outputHeight = inputHeight / sizeHeight;
        int outputWidth = inputWidth / sizeWidth;

        float[,,,] output = new float[batchSize, channels, outputHeight, outputWidth];

        // Create a new array, as we can't work with the out parameter directly in a parallel loop
        (int MaxIndexH, int MaxIndexW)[,,,] maxIndicesArray = new (int MaxIndexH, int MaxIndexW)[batchSize, channels, outputHeight, outputWidth];

        // pre-compute sizes for offsets
        int inputCSize = inputHeight * inputWidth;
        int inputBSize = channels * inputCSize;
        int outputCSize = outputHeight * outputWidth;
        int outputBSize = channels * outputCSize;

        Parallel.ForEach(Partitioner.Create(0, batchSize), range =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

            for (int b = range.Item1; b < range.Item2; b++)
            {
                int inputBIndex = b * inputBSize;
                int outputBIndex = b * outputBSize;
                for (int c = 0; c < channels; c++)
                {
                    int inputCIndex = c * inputCSize;
                    int outputCIndex = c * outputCSize;
                    for (int oh = 0; oh < outputHeight; oh++)
                    {
                        int outputHIndex = oh * outputWidth;
                        for (int ow = 0; ow < outputWidth; ow++)
                        {
                            float maxVal = float.NegativeInfinity;
                            int maxIdxH = -1;
                            int maxIdxW = -1;
                            for (int kh = 0; kh < sizeHeight; kh++)
                            {
                                int ih = oh * sizeHeight + kh;
                                if (ih < inputHeight)
                                {
                                    int inputHIndex = ih * inputWidth;
                                    for (int kw = 0; kw < sizeWidth; kw++)
                                    {
                                        int iw = ow * sizeWidth + kw;
                                        if (iw < inputWidth)
                                        {
                                            // val = input[b, c, ih, iw]
                                            float val = inputSpan[inputBIndex + inputCIndex + inputHIndex + iw];
                                            if (val > maxVal)
                                            {
                                                maxVal = val;
                                                maxIdxH = ih;
                                                maxIdxW = iw;
                                            }
                                        }
                                    }
                                }
                            }
                            // output[b, c, oh, ow] = maxVal;
                            outputSpan[outputBIndex + outputCIndex + outputHIndex + ow] = maxVal;
                            maxIndicesArray[b, c, oh, ow] = (maxIdxH, maxIdxW);
                        }
                    }
                }
            }
        });
        maxIndices = maxIndicesArray;
        return output;
    }

    public override float[,,,] Upsample2DOutput(float[,,,] input, int scaleHeight, int scaleWidth)
    {
        int batchSize = input.GetLength(0);
        int channels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputHeight = inputHeight * scaleHeight;
        int outputWidth = inputWidth * scaleWidth;

        float[,,,] output = new float[batchSize, channels, outputHeight, outputWidth];

        Parallel.ForEach(Partitioner.Create(0, batchSize), range =>
        {
            ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
            Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

            for (int b = range.Item1; b < range.Item2; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < inputHeight; h++)
                    {
                        for (int w = 0; w < inputWidth; w++)
                        {
                            float value = inputSpan[b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + h * inputWidth + w];
                            for (int sh = 0; sh < scaleHeight; sh++)
                            {
                                for (int sw = 0; sw < scaleWidth; sw++)
                                {
                                    outputSpan[b * channels * outputHeight * outputWidth + c * outputHeight * outputWidth + (h * scaleHeight + sh) * outputWidth + (w * scaleWidth + sw)] = value;
                                }
                            }
                        }
                    }
                }
            }
        });
        return output;
    }

    public override float[,,,] Upsample2DInputGradient(float[,,,] input, float[,,,] outputGradient)
    {
        int batchSize = outputGradient.GetLength(0);
        int channels = outputGradient.GetLength(1);
        int outputHeight = outputGradient.GetLength(2);
        int outputWidth = outputGradient.GetLength(3);

        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        Debug.Assert(outputHeight % inputHeight == 0, "Output height must be divisible by input height.");
        Debug.Assert(outputWidth % inputWidth == 0, "Output width must be divisible by input width.");

        int scaleHeight = outputHeight / inputHeight;
        int scaleWidth = outputWidth / inputWidth;

        float[,,,] inputGradient = new float[batchSize, channels, inputHeight, inputWidth];

        Parallel.ForEach(Partitioner.Create(0, batchSize), range =>
        {
            ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
            Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);

            for (int b = range.Item1; b < range.Item2; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < inputHeight; h++)
                    {
                        for (int w = 0; w < inputWidth; w++)
                        {
                            float sum = 0.0f;
                            for (int sh = 0; sh < scaleHeight; sh++)
                            {
                                for (int sw = 0; sw < scaleWidth; sw++)
                                {
                                    sum += outputGradientSpan[b * channels * outputHeight * outputWidth + c * outputHeight * outputWidth + (h * scaleHeight + sh) * outputWidth + (w * scaleWidth + sw)];
                                }
                            }
                            inputGradientSpan[b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + h * inputWidth + w] = sum;
                        }
                    }
                }
            }
        });
        return inputGradient;
    }

    #endregion
}