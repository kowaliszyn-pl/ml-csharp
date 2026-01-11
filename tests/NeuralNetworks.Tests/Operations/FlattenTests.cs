

using NeuralNetworks.Operations.Reshaping;

namespace NeuralNetworks.Operations.Tests;

[TestClass]
public class FlattenTests
{

    [TestMethod]
    public void CalcOutputTest()
    {
        var flatten = new Flatten();
        float[,,,] input = new float[,,,]
        {
            {
                {
                    { 1, 2 },
                    { 3, 4 }
                },
                {
                    { 5, 6 },
                    { 7, 8 }
                }
            },
            {
                {
                    { 11, 12 },
                    { 13, 14 }
                },
                {
                    { 15, 16 },
                    { 17, 18 }
                }
            }
        };

        // Do forward

        float[,] output = flatten.Forward(input, false);

        float[,] expectedOutput = new float[,]
        {
            { 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18 }
        };

        CollectionAssert.AreEqual(expectedOutput, output);

        // Do backward

        float[,,,] inputGradient = flatten.Backward(output);

        CollectionAssert.AreEqual(input, inputGradient);

    }

    [TestMethod]
    public void CalcInputGradientTest()
    {
        var flatten = new Flatten();
        float[,,,] input = new float[,,,]
        {
            {
                {
                    { 1, 2 },
                    { 3, 4 }
                },
                {
                    { 5, 6 },
                    { 7, 8 }
                }
            }
        };

        _ = flatten.Forward(input, false);
        float[,] outputGradient = new float[,]
        {
            { 21, 22, 23, 24, 25, 26, 27, 28 }
        };

        float[,,,] inputGradient = flatten.Backward(outputGradient);

        float[,,,] expectedInputGradient = new float[,,,]
        {
            {
                {
                    { 21, 22 },
                    { 23, 24 }
                },
                {
                    { 25, 26 },
                    { 27, 28 }
                }
            }
        };

        CollectionAssert.AreEqual(expectedInputGradient, inputGradient);
    }

}
