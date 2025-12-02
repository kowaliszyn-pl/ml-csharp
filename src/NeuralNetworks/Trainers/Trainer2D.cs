// Neural Networks in C♯
// File name: Trainer2D.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.Models;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Trainers;

public class Trainer2D : Trainer<float[,], float[,]>
{
    public Trainer2D(Model<float[,], float[,]> model, Optimizer optimizer, SeededRandom random, ILogger<Trainer2D> logger) : base(model, optimizer, random: random, logger: logger)
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
        int trainRows = x.GetLength(0);
        Debug.Assert(trainRows == y.GetLength(0), "Number of samples in x and y do not match.");

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
        => x.GetLength(0);

    // TODO: eval function
}
