// Neural Networks in C♯
// File name: ArrayExtensionsTests.cs
// www.kowaliszyn.pl, 2025 - 2026

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

    private readonly float[,] _input = new float[,] { 
        { 1, 2, 3 }, 
        { -1, -2, -3 } 
    };
    private readonly float[,] _expected = new float[,] {
        { 0.09003057f, 0.24472847f, 0.66524096f },
        { 0.66524096f, 0.24472847f, 0.09003057f }
    };

    [TestMethod]
    public void SoftmaxBasicTest()
    {

        float[,] result = _input.SoftmaxBasic();
        for (int i = 0; i < _expected.GetLength(0); i++)
        {
            for (int j = 0; j < _expected.GetLength(1); j++)
            {
                Assert.AreEqual(_expected[i, j], result[i, j], 1e-6);
            }
        }
    }

    [TestMethod]
    public void SoftmaxStableTest()
    {
        float[,] result = _input.SoftmaxStable();
        for (int i = 0; i < _expected.GetLength(0); i++)
        {
            for (int j = 0; j < _expected.GetLength(1); j++)
            {
                Assert.AreEqual(_expected[i, j], result[i, j], 1e-6);
            }
        }
    }
}
