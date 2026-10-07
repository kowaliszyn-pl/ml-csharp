// Neural Networks in C♯
// File name: PermutationUtilsTests.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Text;

using NeuralNetworks.Core.Utils;

namespace NeuralNetworks.Tests.Core.Utils;

[TestClass]
public class PermutationUtilsTests
{
    private static float[,] CreateMatrix(int rows, int cols)
    {
        float[,] m = new float[rows, cols];
        int v = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                m[i, j] = v++;
            }
        }

        return m;
    }

    private static HashSet<string> RowsAsSet(float[,] m)
    {
        HashSet<string> set = [];
        int rows = m.GetLength(0);
        int cols = m.GetLength(1);
        for (int i = 0; i < rows; i++)
        {
            StringBuilder sb = new();
            for (int j = 0; j < cols; j++)
            {
                sb.Append(m[i, j]).Append(',');
            }

            set.Add(sb.ToString());
        }

        return set;
    }

    [TestMethod]
    public void PermuteInPlace_WithSeededRandom_PreservesRowsAsSet()
    {
        // Arrange
        float[,] source = CreateMatrix(5, 3);
        HashSet<string> expectedRows = RowsAsSet(source);
        Random random = new(42);

        // Act
        source.PermuteInPlace(random);

        // Assert
        CollectionAssert.AreEquivalent(expectedRows.ToList(), RowsAsSet(source).ToList());
    }

    [TestMethod]
    public void PermuteInPlace_WithNullRandom_DoesNotThrow()
    {
        // Arrange
        float[,] source = CreateMatrix(4, 2);
        HashSet<string> expectedRows = RowsAsSet(source);

        // Act
        source.PermuteInPlace();

        // Assert
        CollectionAssert.AreEquivalent(expectedRows.ToList(), RowsAsSet(source).ToList());
    }

    [TestMethod]
    public void Permute_WithSeededRandom_ReturnsNewArrayWithSameRows_SourceUnmodified()
    {
        // Arrange
        float[,] source = CreateMatrix(5, 3);
        float[,] sourceCopy = (float[,])source.Clone();
        Random random = new(7);

        // Act
        float[,] result = source.Permute(random);

        // Assert
        Assert.AreNotSame(source, result);
        CollectionAssert.AreEquivalent(RowsAsSet(sourceCopy).ToList(), RowsAsSet(result).ToList());
        // original array should remain unchanged (since Permute clones before permuting)
        for (int i = 0; i < source.GetLength(0); i++)
        {
            for (int j = 0; j < source.GetLength(1); j++)
            {
                Assert.AreEqual(sourceCopy[i, j], source[i, j]);
            }
        }
    }

    [TestMethod]
    public void Permute_WithNullRandom_DoesNotThrow()
    {
        // Arrange
        float[,] source = CreateMatrix(3, 2);

        // Act
        float[,] result = source.Permute();

        // Assert
        Assert.AreEqual(source.GetLength(0), result.GetLength(0));
        Assert.AreEqual(source.GetLength(1), result.GetLength(1));
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_WithSeededRandom_KeepsXAndYRowsAligned()
    {
        // Arrange
        float[,] x = CreateMatrix(5, 2);
        float[,] y = CreateMatrix(5, 1);
        // y rows identify which original x-row index they correspond to: y[i,0] = i * (2) / 2 pattern.
        // Instead, build y so that y[i,0] equals the first element of x row i, for easy correlation check.
        for (int i = 0; i < 5; i++)
        {
            y[i, 0] = x[i, 0];
        }

        Random random = new(123);

        // Act
        x.PermuteInPlaceTogetherWith(y, random);

        // Assert: after permutation, for each row i, y[i,0] must equal x[i,0] (they moved together)
        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(x[i, 0], y[i, 0]);
        }
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_WithNullRandom_DoesNotThrow()
    {
        // Arrange
        float[,] x = CreateMatrix(4, 2);
        float[,] y = CreateMatrix(4, 1);
        for (int i = 0; i < 4; i++)
        {
            y[i, 0] = x[i, 0];
        }

        // Act
        x.PermuteInPlaceTogetherWith(y);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.AreEqual(x[i, 0], y[i, 0]);
        }
    }

    [TestMethod]
    public void Permute_XY_WithSeededRandom_ReturnsClonedPermutedTuples_SourcesUnmodified()
    {
        // Arrange
        float[,] x = CreateMatrix(5, 2);
        float[,] y = CreateMatrix(5, 1);
        for (int i = 0; i < 5; i++)
        {
            y[i, 0] = x[i, 0];
        }

        float[,] xCopy = (float[,])x.Clone();
        float[,] yCopy = (float[,])y.Clone();
        Random random = new(99);

        // Act
        (float[,] xPermuted, float[,] yPermuted) = PermutationUtils.Permute(x, y, random);

        // Assert
        Assert.AreNotSame(x, xPermuted);
        Assert.AreNotSame(y, yPermuted);
        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(xPermuted[i, 0], yPermuted[i, 0]);
        }

        // originals unchanged
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.AreEqual(xCopy[i, j], x[i, j]);
            }

            Assert.AreEqual(yCopy[i, 0], y[i, 0]);
        }
    }

    [TestMethod]
    public void Permute_XY_WithNullRandom_DoesNotThrow()
    {
        // Arrange
        float[,] x = CreateMatrix(3, 2);
        float[,] y = CreateMatrix(3, 1);

        // Act
        (float[,] xPermuted, float[,] yPermuted) = PermutationUtils.Permute(x, y);

        // Assert
        Assert.AreEqual(3, xPermuted.GetLength(0));
        Assert.AreEqual(3, yPermuted.GetLength(0));
    }

    [TestMethod]
    public void PermuteTogetherWith_WithSeededRandom_KeepsXAndYRowsAligned_InPlace()
    {
        // Arrange
        float[,] x = CreateMatrix(5, 2);
        float[,] y = CreateMatrix(5, 1);
        for (int i = 0; i < 5; i++)
        {
            y[i, 0] = x[i, 0];
        }

        Random random = new(55);

        // Act
        PermutationUtils.PermuteTogetherWith(x, y, random);

        // Assert
        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(x[i, 0], y[i, 0]);
        }
    }

    [TestMethod]
    public void PermuteTogetherWith_WithNullRandom_DoesNotThrow()
    {
        // Arrange
        float[,] x = CreateMatrix(4, 2);
        float[,] y = CreateMatrix(4, 1);
        for (int i = 0; i < 4; i++)
        {
            y[i, 0] = x[i, 0];
        }

        // Act
        PermutationUtils.PermuteTogetherWith(x, y);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.AreEqual(x[i, 0], y[i, 0]);
        }
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_Float3DWithFloat2D_WithSeededRandom_KeepsXAndYRowsAligned()
    {
        // Arrange
        float[,,] x = new float[5, 2, 3];
        float[,] y = CreateMatrix(5, 1);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    x[i, j, k] = (i * 100) + (j * 10) + k;
                }
            }

            y[i, 0] = i;
        }

        Random random = new(321);

        // Act
        x.PermuteInPlaceTogetherWith(y, random);

        // Assert: for each row i, x[i,0,0] / 100 must equal y[i,0] (they moved together)
        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(y[i, 0] * 100, x[i, 0, 0]);
        }
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_Float3DWithFloat2D_WithNullRandom_DoesNotThrow()
    {
        // Arrange
        float[,,] x = new float[4, 2, 2];
        float[,] y = CreateMatrix(4, 1);
        for (int i = 0; i < 4; i++)
        {
            x[i, 0, 0] = i;
            y[i, 0] = i;
        }

        // Act
        x.PermuteInPlaceTogetherWith(y);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.AreEqual(y[i, 0], x[i, 0, 0]);
        }
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_Float4DWithFloat2D_WithSeededRandom_KeepsXAndYRowsAligned()
    {
        // Arrange
        float[,,,] x = new float[5, 2, 2, 2];
        float[,] y = CreateMatrix(5, 1);
        for (int i = 0; i < 5; i++)
        {
            x[i, 0, 0, 0] = i;
            y[i, 0] = i;
        }

        Random random = new(654);

        // Act
        x.PermuteInPlaceTogetherWith(y, random);

        // Assert
        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(y[i, 0], x[i, 0, 0, 0]);
        }
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_Float4DWithFloat2D_WithNullRandom_DoesNotThrow()
    {
        // Arrange
        float[,,,] x = new float[4, 2, 2, 2];
        float[,] y = CreateMatrix(4, 1);
        for (int i = 0; i < 4; i++)
        {
            x[i, 0, 0, 0] = i;
            y[i, 0] = i;
        }

        // Act
        x.PermuteInPlaceTogetherWith(y);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.AreEqual(y[i, 0], x[i, 0, 0, 0]);
        }
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_Float4DWithFloat4D_WithSeededRandom_KeepsXAndYRowsAligned()
    {
        // Arrange
        float[,,,] x = new float[5, 2, 2, 2];
        float[,,,] y = new float[5, 1, 1, 1];
        for (int i = 0; i < 5; i++)
        {
            x[i, 0, 0, 0] = i;
            y[i, 0, 0, 0] = i;
        }

        Random random = new(987);

        // Act
        x.PermuteInPlaceTogetherWith(y, random);

        // Assert
        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(y[i, 0, 0, 0], x[i, 0, 0, 0]);
        }
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_Float4DWithFloat4D_WithNullRandom_DoesNotThrow()
    {
        // Arrange
        float[,,,] x = new float[4, 2, 2, 2];
        float[,,,] y = new float[4, 1, 1, 1];
        for (int i = 0; i < 4; i++)
        {
            x[i, 0, 0, 0] = i;
            y[i, 0, 0, 0] = i;
        }

        // Act
        x.PermuteInPlaceTogetherWith(y);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.AreEqual(y[i, 0, 0, 0], x[i, 0, 0, 0]);
        }
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_IntWithFloat2D_WithSeededRandom_KeepsXAndYRowsAligned()
    {
        // Arrange
        int[,] x = new int[5, 2];
        float[,] y = CreateMatrix(5, 1);
        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = i;
            y[i, 0] = i;
        }

        Random random = new(159);

        // Act
        x.PermuteInPlaceTogetherWith(y, random);

        // Assert
        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(y[i, 0], x[i, 0]);
        }
    }

    [TestMethod]
    public void PermuteInPlaceTogetherWith_IntWithFloat2D_WithNullRandom_DoesNotThrow()
    {
        // Arrange
        int[,] x = new int[4, 2];
        float[,] y = CreateMatrix(4, 1);
        for (int i = 0; i < 4; i++)
        {
            x[i, 0] = i;
            y[i, 0] = i;
        }

        // Act
        x.PermuteInPlaceTogetherWith(y);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.AreEqual(y[i, 0], x[i, 0]);
        }
    }

    [TestMethod]
    public void AllMethodsPermutesInTheSameWay()
    {
        // Arrange
        const int rows = 5;
        const int cols = 3;
        const int dim = 2;
        const int seed = 19690914;

        int[,] a1 = new int[rows, cols];
        float[,] a2 = new float[rows, cols];
        float[,,] a3 = new float[rows, cols, dim];
        float[,,,] a4 = new float[rows, cols, dim, dim];

        // Fill arrays with identifiable values
        for (int i = 0; i < rows; i++)
        {
            a1[i, 0] = i;
            a2[i, 0] = i;
            a3[i, 0, 0] = i;
            a4[i, 0, 0, 0] = i;
        }

        // Act
        a1.PermuteInPlace(new(seed));
        a2.PermuteInPlace(new(seed));
        a3.PermuteInPlace(new(seed));
        a4.PermuteInPlace(new(seed));

        // Assert
        Assert.IsTrue(a1[0, 0] == a2[0, 0] && a1[0, 0] == a3[0, 0, 0] && a1[0, 0] == a4[0, 0, 0, 0] && a1[0, 0] == 4);
        Assert.IsTrue(a1[1, 0] == a2[1, 0] && a1[1, 0] == a3[1, 0, 0] && a1[1, 0] == a4[1, 0, 0, 0] && a1[1, 0] == 1);
        Assert.IsTrue(a1[2, 0] == a2[2, 0] && a1[2, 0] == a3[2, 0, 0] && a1[2, 0] == a4[2, 0, 0, 0] && a1[2, 0] == 0);
        Assert.IsTrue(a1[3, 0] == a2[3, 0] && a1[3, 0] == a3[3, 0, 0] && a1[3, 0] == a4[3, 0, 0, 0] && a1[3, 0] == 2);
        Assert.IsTrue(a1[4, 0] == a2[4, 0] && a1[4, 0] == a3[4, 0, 0] && a1[4, 0] == a4[4, 0, 0, 0] && a1[4, 0] == 3);
    }
}