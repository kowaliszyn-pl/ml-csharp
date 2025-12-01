// Machine Learning Utils
// File name: Conv2DOperation.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations;

// TODO: strides, custom padding, dilation

/// <summary>
/// Dimensions of the input are: [batch, channels, height, width]
/// Dimensions of the param array are: [channels, filters, kernelSize, kernelSize]
/// Padding is assumed to be the same on all sides = kernelSize / 2
/// </summary>
/// <param name="weights"></param>
public class Conv2D(float[,,,] weights) : ParamOperation4D<float[,,,]>(weights)
{
    
    protected override float[,,,] CalcOutput(bool inference)
    {
        int batchSize = Input.GetLength(0);
        int inputChannels = Input.GetLength(1);
        int inputHeight = Input.GetLength(2);
        int inputWidth = Input.GetLength(3);

        int outputChannels = Param.GetLength(1);
        int kernelSize = Param.GetLength(2);

        Debug.Assert(Param.GetLength(0) == inputChannels);
        Debug.Assert(kernelSize == Param.GetLength(3));

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
                                        sum += Input[b, ic, ih, iw] * Param[ic, oc, kh, kw];
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

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
    {
        int batchSize = outputGradient.GetLength(0);
        int inputChannels = Input.GetLength(1);
        int inputHeight = Input.GetLength(2);
        int inputWidth = Input.GetLength(3);

        int outputChannels = outputGradient.GetLength(1);
        int kernelSize = Param.GetLength(2);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        Debug.Assert(Param.GetLength(0) == inputChannels);
        Debug.Assert(kernelSize == Param.GetLength(3));

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
                                        sum += outputGradient[b, oc, oh, ow] * Param[ic, oc, kh, kw];
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

    protected override float[,,,] CalcParamGradient(float[,,,] outputGradient)
    {
        int batchSize = outputGradient.GetLength(0);
        int inputChannels = Input.GetLength(1);
        int inputHeight = Input.GetLength(2);
        int inputWidth = Input.GetLength(3);

        int outputChannels = outputGradient.GetLength(1);
        int kernelSize = Param.GetLength(2);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        Debug.Assert(Param.GetLength(0) == inputChannels);
        Debug.Assert(kernelSize == Param.GetLength(3));

        int padding = kernelSize / 2;

        float[,,,] paramGradient = new float[inputChannels, outputChannels, kernelSize, kernelSize];

        for (int b = 0; b < batchSize; b++)
        {
            for (int ic = 0; ic < inputChannels; ic++)
            {
                for (int oc = 0; oc < outputChannels; oc++)
                {
                    for (int kh = 0; kh < kernelSize; kh++)
                    {
                        for (int kw = 0; kw < kernelSize; kw++)
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
                                        sum += outputGradient[b, oc, oh, ow] * Input[b, ic, ih, iw];
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

    public override void UpdateParams(Layer? layer, Optimizer optimizer)
        => optimizer.Update(layer, Param, ParamGradient);

    protected override void EnsureSameShapeForParam(float[,,,]? param, float[,,,] paramGradient) 
        => EnsureSameShape(param, paramGradient);

    public override int GetParamCount() 
        => Param.Length;
}
