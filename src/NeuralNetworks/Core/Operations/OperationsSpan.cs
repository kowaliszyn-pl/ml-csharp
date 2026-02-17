// Neural Networks in C♯
// File name: OperationsSpan.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NeuralNetworks.Core.Operations;

public class OperationsSpan : OperationsArray
{
    #region Backend Management

    public override OperationBackendType BackendType => OperationBackendType.CpuSpans;

    #endregion

    #region Loss Functions

    public override float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        float loss = 0f;
        int batchSize = predicted.GetLength(0);
        int numClasses = predicted.GetLength(1);

        ReadOnlySpan<float> predictedSpan = MemoryMarshal.CreateReadOnlySpan(ref predicted[0, 0], predicted.Length);
        ReadOnlySpan<float> targetSpan = MemoryMarshal.CreateReadOnlySpan(ref target[0, 0], target.Length);

        for (int i = 0; i < targetSpan.Length; i++)
        {
            float p = Math.Clamp(predictedSpan[i], eps, 1 - eps);
            loss += targetSpan[i] * MathF.Log(p);
        }

        return -loss / (batchSize * numClasses);
    }

    public override float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        int batchSize = predicted.GetLength(0);
        int numClasses = predicted.GetLength(1);
        float[,] gradient = new float[batchSize, numClasses];

        ReadOnlySpan<float> predictedSpan = MemoryMarshal.CreateReadOnlySpan(ref predicted[0, 0], predicted.Length);
        ReadOnlySpan<float> targetSpan = MemoryMarshal.CreateReadOnlySpan(ref target[0, 0], target.Length);
        Span<float> gradientSpan = MemoryMarshal.CreateSpan(ref gradient[0, 0], gradient.Length);

        for (int i = 0; i < gradientSpan.Length; i++)
        {
            gradientSpan[i] = (predictedSpan[i] - targetSpan[i]) / batchSize;
        }

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

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
        Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

        for (int i = 0; i < inputSpan.Length; i++)
        {
            float value = inputSpan[i];
            outputSpan[i] = value >= 0 ? value * beta : value * alpha;
        }

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

        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
        Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);

        for (int i = 0; i < inputGradientSpan.Length; i++)
        {
            inputGradientSpan[i] = inputSpan[i] > 0 ? outputGradientSpan[i] * beta : outputGradientSpan[i] * alfa;
        }
        return inputGradient;
    }

    public override float[,,,] ReLUOutput(float[,,,] input, float beta = 1)
    {
        int dim1 = input.GetLength(0);
        int dim2 = input.GetLength(1);
        int dim3 = input.GetLength(2);
        int dim4 = input.GetLength(3);
        float[,,,] output = new float[dim1, dim2, dim3, dim4];
        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
        Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);
        for (int i = 0; i < inputSpan.Length; i++)
        {
            float value = inputSpan[i];
            outputSpan[i] = value >= 0 ? value * beta : 0f;
        }
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
        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
        Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);
        for (int i = 0; i < inputGradientSpan.Length; i++)
        {
            inputGradientSpan[i] = inputSpan[i] > 0 ? outputGradientSpan[i] * beta : 0f;
        }
        return inputGradient;
    }

    public override float[,,,] TanhOutput(float[,,,] source)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        Debug.Assert(dim1 > 0 && dim2 > 0 && dim3 > 0 && dim4 > 0, "All dimensions must be greater than zero.");

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        ReadOnlySpan<float> sourceSpan = MemoryMarshal.CreateReadOnlySpan(ref source[0, 0, 0, 0], source.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref res[0, 0, 0, 0], res.Length);

        for (int i = 0; i < resSpan.Length; i++)
        {
            resSpan[i] = MathF.Tanh(sourceSpan[i]);
        }

        return res;
    }

    public override float[,,,] TanhInputGradient(float[,,,] outputGradient, float[,,,] output)
    {
        int dim0 = outputGradient.GetLength(0);
        int dim1 = outputGradient.GetLength(1);
        int dim2 = outputGradient.GetLength(2);
        int dim3 = outputGradient.GetLength(3);

        Debug.Assert(dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(output.GetLength(0) == dim0 && output.GetLength(1) == dim1 && output.GetLength(2) == dim2 && output.GetLength(3) == dim3, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,,,] inputGradient = new float[dim0, dim1, dim2, dim3];

        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
        ReadOnlySpan<float> outputSpan = MemoryMarshal.CreateReadOnlySpan(ref output[0, 0, 0, 0], output.Length);
        Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);

        for (int i = 0; i < inputGradientSpan.Length; i++)
        {
            float y = outputSpan[i];
            inputGradientSpan[i] = outputGradientSpan[i] * (1f - (y * y));
        }

        return inputGradient;
    }

    #endregion

    #region Parametric Operations

    #region Convolution 2D

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

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0, 0, 0], weights.Length);
        Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

        // pre-compute sizes for offsets
        int outputCSize = outputHeight * outputWidth;
        int outputBSize = outputChannels * outputCSize;
        int inputCSize = inputHeight * inputWidth;
        int inputBSize = inputChannels * inputCSize;
        int weightsOutputCSize = kernelHeight * kernelWidth;
        int weightsCSize = outputChannels * weightsOutputCSize;

        for (int b = 0; b < batchSize; b++) // ~100
        {
            int inputBIndex = b * inputBSize;
            int outputBIndex = b * outputBSize;
            for (int oc = 0; oc < outputChannels; oc++) // ~32
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
                                int ih = kh * dilatationHeight + ohMinusPad;
                                if (ih >= 0 && ih < inputHeight)
                                {
                                    int inputHIndex = ih * inputWidth;
                                    for (int kw = 0; kw < kernelWidth; kw++) // ~3
                                    {
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

        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0, 0, 0], weights.Length);
        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
        Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);

        // pre-compute sizes for offsets
        int outputGradientCSize = outputGradientHeight * outputGradientWidth;
        int outputGradientBSize = outputGradientChannels * outputGradientCSize;
        int weightsOutputCSize = kernelHeight * kernelWidth;
        int weightsInputCSize = outputGradientChannels * weightsOutputCSize;
        int inputGradientCSize = inputHeight * inputWidth;
        int inputGradientBSize = inputChannels * inputGradientCSize;

        for (int b = 0; b < batchSize; b++)
        {
            int outputGradientBIndex = b * outputGradientBSize;
            int inputGradientBIndex = b * inputGradientBSize;
            for (int ic = 0; ic < inputChannels; ic++)
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
                                if (oh >= 0 && oh < outputGradientHeight && remH == 0)
                                {
                                    int weightsKernelHIndex = kh * kernelWidth;
                                    int outputHIndex = oh * outputGradientWidth;
                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int ow = Math.DivRem(iwPlusPad - kw * dilatationWidth, strideWidth, out int remW);
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

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
        Span<float> paramGradientSpan = MemoryMarshal.CreateSpan(ref paramGradient[0, 0, 0, 0], paramGradient.Length);

        // pre-compute sizes for offsets
        int outputGradientOutputCSize = outputGradientHeight * outputGradientWidth;
        int outputGradientBSize = outputGradientChannels * outputGradientOutputCSize;
        int inputCSize = inputHeight * inputWidth;
        int inputBSize = inputChannels * inputCSize;
        int paramGradientOutputCSize = kernelHeight * kernelWidth;
        int paramGradientInputCSize = outputGradientChannels * paramGradientOutputCSize;

        for (int b = 0; b < batchSize; b++)
        {
            int outputGradientBIndex = b * outputGradientBSize;
            int inputBIndex = b * inputBSize;
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
                                int ih = oh * strideHeight + khMinusPad;
                                if (ih >= 0 && ih < inputHeight)
                                {
                                    int inputHIndex = ih * inputWidth;
                                    int outputGradientHIndex = oh * outputGradientWidth;
                                    for (int ow = 0; ow < outputGradientWidth; ow++)
                                    {
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
                            // paramGradient[ic, oc, kh, kw] += sum;
                            paramGradientSpan[paramGradientInputCIndex + paramGradientOutputCIndex + paramGradientKernelHIndex + kw] += sum;
                        }
                    }
                }
            }
        }

        return paramGradient;
    }

    #endregion

    #region WeightMultiply

    public override float[,] WeightMultiplyOutput(float[,] input, float[,] weights)
    {
        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = weights.GetLength(1);

        Debug.Assert(weights.GetLength(0) == inputFeatures, "Input features must match weight input dimension.");

        float[,] output = new float[batchSize, outputFeatures];

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0], input.Length);
        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0], weights.Length);
        Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0], output.Length);

        for (int b = 0; b < batchSize; b++)
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

        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0], weights.Length);
        Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0], inputGradient.Length);

        for (int b = 0; b < batchSize; b++)
        {
            int outputGradientBIndex = b * outputFeatures;
            int inputGradientBIndex = b * inputFeatures;
            for (int inputFeature = 0; inputFeature < inputFeatures; inputFeature++)
            {
                float sum = 0f;
                for (int outputFeature = 0; outputFeature < outputFeatures; outputFeature++)
                {
                    // sum += outputGradient[b, outputFeature] * weights[inputFeature, outputFeature];
                    sum += outputGradientSpan[outputGradientBIndex + outputFeature] * weightsSpan[inputFeature * outputFeatures + outputFeature];
                }
                // inputGradient[b, inputFeature] = sum;
                inputGradientSpan[inputGradientBIndex + inputFeature] = sum;
            }
        }

        return inputGradient;
    }

    public override float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient)
    {
        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = outputGradient.GetLength(1);

        Debug.Assert(outputGradient.GetLength(0) == batchSize, "Batch size of output gradient must match batch size of input.");

        float[,] paramGradient = new float[inputFeatures, outputFeatures];

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0], input.Length);
        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
        Span<float> paramGradientSpan = MemoryMarshal.CreateSpan(ref paramGradient[0, 0], paramGradient.Length);

        for (int ifeature = 0; ifeature < inputFeatures; ifeature++)
        {
            int paramGradientInputFeatureIndex = ifeature * outputFeatures;
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
        return paramGradient;
    }

    #endregion

    #endregion

    #region Transformations

    public override float[,] Flatten(float[,,,] source)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        Debug.Assert(dim1 > 0 && dim2 > 0 && dim3 > 0 && dim4 > 0, "All dimensions must be greater than zero.");

        float[,] res = new float[dim1, dim2 * dim3 * dim4];

        ReadOnlySpan<float> sourceSpan = MemoryMarshal.CreateReadOnlySpan(ref source[0, 0, 0, 0], source.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref res[0, 0], res.Length);

        sourceSpan.CopyTo(resSpan);

        return res;
    }

    public override float[,,,] Unflatten(float[,] source, float[,,,] targetSize)
    {
        int dim1 = targetSize.GetLength(0);
        int dim2 = targetSize.GetLength(1);
        int dim3 = targetSize.GetLength(2);
        int dim4 = targetSize.GetLength(3);

        Debug.Assert(dim1 > 0 && dim2 > 0 && dim3 > 0 && dim4 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(source.GetLength(0) == dim1 && source.GetLength(1) == dim2 * dim3 * dim4, "Source shape does not match target size for unflattening.");

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        ReadOnlySpan<float> sourceSpan = MemoryMarshal.CreateReadOnlySpan(ref source[0, 0], source.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref res[0, 0, 0, 0], res.Length);

        sourceSpan.CopyTo(resSpan);

        return res;
    }

    #endregion
}
