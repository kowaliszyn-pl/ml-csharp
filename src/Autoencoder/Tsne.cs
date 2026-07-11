// Neural Networks in C♯
// File name: Tsne.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace Autoencoder;

/// <summary>
/// t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation for dimensionality reduction.
/// </summary>
public class Tsne
{
    private readonly Random _random;
    private readonly int _outputDimensions;
    private readonly double _perplexity;
    private readonly int _maxIterations;
    private readonly double _learningRate;

    public Tsne(int outputDimensions = 2, double perplexity = 30.0, int maxIterations = 1000, double learningRate = 200.0, int randomSeed = 42)
    {
        _outputDimensions = outputDimensions;
        _perplexity = perplexity;
        _maxIterations = maxIterations;
        _learningRate = learningRate;
        _random = new Random(randomSeed);
    }

    public double[,] FitTransform(float[,] data, Action<int, double>? progressCallback = null)
    {
        int n = data.GetLength(0);
        int d = data.GetLength(1);

        Console.WriteLine($"Computing pairwise distances for {n} points...");
        double[,] distances = ComputeEuclideanDistances(data);

        Console.WriteLine("Computing P (high-dimensional affinities)...");
        double[,] P = ComputePairwiseAffinities(distances);

        Console.WriteLine("Initializing low-dimensional embedding...");
        double[,] Y = InitializeEmbedding(n, _outputDimensions);

        Console.WriteLine("Running gradient descent optimization...");
        double[,] gains = CreateMatrix(n, _outputDimensions, 1.0);
        double[,] yGrads = CreateMatrix(n, _outputDimensions, 0.0);
        double[,] yIncs = CreateMatrix(n, _outputDimensions, 0.0);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Compute Q (low-dimensional affinities)
            double[,] Q = ComputeLowDimensionalAffinities(Y);

            // Compute gradient
            double[,] gradient = ComputeGradient(P, Q, Y);

            // Update with momentum and adaptive learning rate
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < _outputDimensions; j++)
                {
                    // Adaptive learning rate (gain)
                    if (Math.Sign(gradient[i, j]) != Math.Sign(yIncs[i, j]))
                        gains[i, j] += 0.2;
                    else
                        gains[i, j] *= 0.8;

                    if (gains[i, j] < 0.01)
                        gains[i, j] = 0.01;

                    // Momentum and update
                    double momentum = iter < 250 ? 0.5 : 0.8;
                    yIncs[i, j] = momentum * yIncs[i, j] - _learningRate * gains[i, j] * gradient[i, j];
                    Y[i, j] += yIncs[i, j];
                }
            }

            // Zero-mean the solution
            ZeroMean(Y);

            if ((iter + 1) % 50 == 0)
            {
                double error = ComputeKLDivergence(P, Q);
                Console.WriteLine($"Iteration {iter + 1}/{_maxIterations}, KL divergence = {error:F4}");
                progressCallback?.Invoke(iter + 1, error);
            }
        }

        return Y;
    }

    private double[,] ComputeEuclideanDistances(float[,] data)
    {
        int n = data.GetLength(0);
        int d = data.GetLength(1);
        double[,] distances = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double sum = 0;
                for (int k = 0; k < d; k++)
                {
                    double diff = data[i, k] - data[j, k];
                    sum += diff * diff;
                }
                distances[i, j] = Math.Sqrt(sum);
                distances[j, i] = distances[i, j];
            }
        }

        return distances;
    }

    private double[,] ComputePairwiseAffinities(double[,] distances)
    {
        int n = distances.GetLength(0);
        double[,] P = new double[n, n];

        // Compute conditional probabilities using Gaussian kernel with binary search for sigma
        for (int i = 0; i < n; i++)
        {
            double beta = FindBeta(distances, i, _perplexity);

            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    P[i, j] = Math.Exp(-distances[i, j] * distances[i, j] * beta);
                    sum += P[i, j];
                }
            }

            // Normalize
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                    P[i, j] /= sum;
            }
        }

        // Symmetrize and normalize
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                P[i, j] = (P[i, j] + P[j, i]) / (2.0 * n);
            }
        }

        return P;
    }

    private double FindBeta(double[,] distances, int i, double targetPerplexity)
    {
        int n = distances.GetLength(0);
        double betaMin = double.NegativeInfinity;
        double betaMax = double.PositiveInfinity;
        double beta = 1.0;
        const double tolerance = 1e-5;
        const int maxIterations = 50;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            double sumP = 0;
            double sumDsqP = 0;

            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    double pji = Math.Exp(-distances[i, j] * distances[i, j] * beta);
                    sumP += pji;
                    sumDsqP += distances[i, j] * distances[i, j] * pji;
                }
            }

            double entropy = Math.Log(sumP) + beta * sumDsqP / sumP;
            double perplexity = Math.Exp(entropy);
            double perplexityDiff = perplexity - targetPerplexity;

            if (Math.Abs(perplexityDiff) < tolerance)
                break;

            if (perplexityDiff > 0)
            {
                betaMin = beta;
                beta = double.IsPositiveInfinity(betaMax) ? beta * 2 : (beta + betaMax) / 2;
            }
            else
            {
                betaMax = beta;
                beta = double.IsNegativeInfinity(betaMin) ? beta / 2 : (beta + betaMin) / 2;
            }
        }

        return beta;
    }

    private double[,] ComputeLowDimensionalAffinities(double[,] Y)
    {
        int n = Y.GetLength(0);
        double[,] Q = new double[n, n];
        double sum = 0;

        // Compute Student t-distribution (df=1)
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double distSq = 0;
                for (int k = 0; k < _outputDimensions; k++)
                {
                    double diff = Y[i, k] - Y[j, k];
                    distSq += diff * diff;
                }

                double qij = 1.0 / (1.0 + distSq);
                Q[i, j] = qij;
                Q[j, i] = qij;
                sum += 2 * qij;
            }
        }

        // Normalize
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Q[i, j] = Math.Max(Q[i, j] / sum, 1e-12);
            }
        }

        return Q;
    }

    private double[,] ComputeGradient(double[,] P, double[,] Q, double[,] Y)
    {
        int n = Y.GetLength(0);
        double[,] gradient = new double[n, _outputDimensions];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    double mult = (P[i, j] - Q[i, j]);
                    double distSq = 0;
                    for (int k = 0; k < _outputDimensions; k++)
                    {
                        double diff = Y[i, k] - Y[j, k];
                        distSq += diff * diff;
                    }
                    double weight = mult / (1.0 + distSq);

                    for (int k = 0; k < _outputDimensions; k++)
                    {
                        gradient[i, k] += 4 * weight * (Y[i, k] - Y[j, k]);
                    }
                }
            }
        }

        return gradient;
    }

    private double ComputeKLDivergence(double[,] P, double[,] Q)
    {
        int n = P.GetLength(0);
        double kl = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != j && P[i, j] > 1e-12)
                {
                    kl += P[i, j] * Math.Log(P[i, j] / Q[i, j]);
                }
            }
        }

        return kl;
    }

    private double[,] InitializeEmbedding(int n, int dimensions)
    {
        double[,] Y = new double[n, dimensions];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                Y[i, j] = (_random.NextDouble() - 0.5) * 0.0001;
            }
        }

        return Y;
    }

    private static double[,] CreateMatrix(int rows, int cols, double initialValue)
    {
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = initialValue;
            }
        }
        return matrix;
    }

    private static void ZeroMean(double[,] Y)
    {
        int n = Y.GetLength(0);
        int d = Y.GetLength(1);

        for (int j = 0; j < d; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
            {
                mean += Y[i, j];
            }
            mean /= n;

            for (int i = 0; i < n; i++)
            {
                Y[i, j] -= mean;
            }
        }
    }
}