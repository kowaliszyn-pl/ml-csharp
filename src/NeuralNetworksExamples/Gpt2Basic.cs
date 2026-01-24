// Neural Networks in C?
// File name: Gpt2Basic.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Linq;

using NeuralNetworks.Core;
using NeuralNetworks.Transformers.Gpt2;

namespace NeuralNetworksExamples;

internal static class Gpt2Basic
{
    private const float InitScale = 0.02f;

    public static void Run()
    {
        Gpt2Config config = new(
            vocabularySize: 32,
            contextSize: 32,
            embeddingSize: 16,
            headCount: 4,
            layerCount: 2);

        SeededRandom random = new(42);
        Gpt2Parameters parameters = RandomParameterFactory.Create(config, random);
        Gpt2Model model = new(config);

        List<int> prompt = new() { 0, 4, 7, 2 };
        IReadOnlyList<int> generated = model.Generate(prompt, parameters, tokensToGenerate: 5);

        Console.WriteLine("GPT-2 toy run (ids only): " + string.Join(", ", generated));
    }

    private static class RandomParameterFactory
    {
        public static Gpt2Parameters Create(Gpt2Config config, Random random)
        {
            float[,] tokenEmbeddings = SampleMatrix(random, config.VocabularySize, config.EmbeddingSize);
            float[,] positionalEmbeddings = SampleMatrix(random, config.ContextSize, config.EmbeddingSize);

            TransformerBlockParameters[] blocks = new TransformerBlockParameters[config.LayerCount];
            for (int i = 0; i < blocks.Length; i++)
            {
                LinearWeights cAttn = SampleLinear(random, config.EmbeddingSize, config.EmbeddingSize * 3);
                LinearWeights cProj = SampleLinear(random, config.EmbeddingSize, config.EmbeddingSize);

                LinearWeights up = SampleLinear(random, config.EmbeddingSize, config.EmbeddingSize * 4);
                LinearWeights down = SampleLinear(random, config.EmbeddingSize * 4, config.EmbeddingSize);

                LayerNormParameters ln1 = CreateLayerNorm(config.EmbeddingSize);
                LayerNormParameters ln2 = CreateLayerNorm(config.EmbeddingSize);

                blocks[i] = new TransformerBlockParameters(
                    new MultiHeadAttentionParameters(cAttn, cProj),
                    new FeedForwardParameters(up, down),
                    ln1,
                    ln2);
            }

            LayerNormParameters finalLn = CreateLayerNorm(config.EmbeddingSize);
            return new Gpt2Parameters(tokenEmbeddings, positionalEmbeddings, blocks, finalLn);
        }

        private static LayerNormParameters CreateLayerNorm(int dimension)
        {
            float[] gamma = Enumerable.Repeat(1f, dimension).ToArray();
            float[] beta = new float[dimension];
            return new LayerNormParameters(gamma, beta);
        }

        private static LinearWeights SampleLinear(Random random, int inputDim, int outputDim)
        {
            float[,] weights = SampleMatrix(random, inputDim, outputDim);
            float[] bias = new float[outputDim];
            for (int i = 0; i < bias.Length; i++)
                bias[i] = SampleValue(random);
            return new LinearWeights(weights, bias);
        }

        private static float[,] SampleMatrix(Random random, int rows, int cols)
        {
            float[,] matrix = new float[rows, cols];
            for (int row = 0; row < rows; row++)
                for (int col = 0; col < cols; col++)
                    matrix[row, col] = SampleValue(random);
            return matrix;
        }

        private static float SampleValue(Random random)
        {
            double uniform = random.NextDouble();
            return (float)(uniform * 2 * InitScale - InitScale);
        }
    }
}
