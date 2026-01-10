// Machine Learning Utils
// File name: ArrayTests.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.Core.Tests;

[TestClass]
public class ArrayExtensionsTests
{
    [TestMethod]
    public void MultiplyTest2D()
    {
        float[,] matrix = new float[,] { { 1, 2 }, { 3, 4 } };
        float[,] result = matrix.Multiply(2);
        Assert.AreEqual(2f, result[0, 0]);
        Assert.AreEqual(8f, result[1, 1]);
    }

    [TestMethod]
    public void MultiplyTest4D()
    {
        float[,,,] matrix = new float[,,,] {
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
                    { 9, 10 },
                    { 11, 12 }
                },
                {
                    { 13, 14 },
                    { 15, 16 }
                }
            }
        };
        float[,,,] result = matrix.Multiply(2);

        CollectionAssert.AreEqual(new float[,,,] {
            {
                {
                    { 2, 4 },
                    { 6, 8 }
                },
                {
                    { 10, 12 },
                    { 14, 16 }
                }
            },
            {
                {
                    { 18, 20 },
                    { 22, 24 }
                },
                {
                    { 26, 28 },
                    { 30, 32 }
                }
            }
        }, result);
    }
}
