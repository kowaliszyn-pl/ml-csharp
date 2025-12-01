// Machine Learning Utils
// File name: Array2DTrainer.cs
// Code It Yourself with .NET, 2024

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.Models;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Trainers;

public class Trainer2D : Trainer<float[,], float[,]>
{
    public Trainer2D(NeuralNetwork<float[,], float[,]> neuralNetwork, Optimizer optimizer, SeededRandom random, ILogger<Trainer2D> logger) : base(neuralNetwork, optimizer, random: random, logger: logger)
    {
    }

    /// <summary>
    /// Generates batches of input and output matrices.
    /// </summary>
    /// <param name="x">The input matrix.</param>
    /// <param name="y">The output matrix.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>An enumerable of batches.</returns>
    protected override IEnumerable<(float[,] xBatch, float[,] yBatch)> GenerateBatches(float[,] x, float[,] y, int batchSize)
    {
        int trainRows = x.GetLength((int)Dimension.Rows);
#if DEBUG
        if (trainRows != y.GetLength((int)Dimension.Rows))
        {
            throw new ArgumentException("Number of samples in x and y do not match.");
        }
#endif
        for (int batchStart = 0; batchStart < trainRows; batchStart += batchSize)
        {
            int effectiveBatchSize = Math.Min(batchSize, trainRows - batchStart);
            int batchEnd = effectiveBatchSize + batchStart;
            float[,] xBatch = x.GetRows(batchStart..batchEnd);
            float[,] yBatch = y.GetRows(batchStart..batchEnd);
            yield return (xBatch, yBatch);
        }
    }

    protected override (float[,], float[,]) PermuteData(float[,] x, float[,] y, Random random) 
        => ArrayUtils.PermuteData(x, y, random);

    protected override float GetRows(float[,] x) 
        => x.GetLength((int)Dimension.Rows);

    // TODO: eval function
}
