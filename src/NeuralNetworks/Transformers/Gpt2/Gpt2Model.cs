// Neural Networks in C?
// File name: Gpt2Model.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Transformers.Gpt2;

public sealed class Gpt2Config
{
    public Gpt2Config(int vocabularySize, int contextSize, int embeddingSize, int headCount, int layerCount)
    {
        if (vocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (contextSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(contextSize));
        if (embeddingSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingSize));
        if (headCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(headCount));
        if (layerCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(layerCount));

        if (embeddingSize % headCount != 0)
            throw new ArgumentException("Embedding size must be divisible by number of heads.", nameof(embeddingSize));

        VocabularySize = vocabularySize;
        ContextSize = contextSize;
        EmbeddingSize = embeddingSize;
        HeadCount = headCount;
        LayerCount = layerCount;
    }

    public int VocabularySize { get; }
    public int ContextSize { get; }
    public int EmbeddingSize { get; }
    public int HeadCount { get; }
    public int LayerCount { get; }
    public int HeadSize => EmbeddingSize / HeadCount;
}

public sealed class LinearWeights
{
    public LinearWeights(float[,] weights, float[] bias)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(bias);

        if (bias.Length != weights.GetLength(1))
            throw new ArgumentException("Bias length must match number of columns in the weight matrix.", nameof(bias));

        Weights = weights;
        Bias = bias;
    }

    public float[,] Weights { get; }
    public float[] Bias { get; }
}

public sealed class LayerNormParameters
{
    public LayerNormParameters(float[] gamma, float[] beta, float epsilon = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(gamma);
        ArgumentNullException.ThrowIfNull(beta);
        if (gamma.Length != beta.Length)
            throw new ArgumentException("Gamma and beta must have the same length.");
        if (epsilon <= 0f)
            throw new ArgumentOutOfRangeException(nameof(epsilon));

        Gamma = gamma;
        Beta = beta;
        Epsilon = epsilon;
    }

    public float[] Gamma { get; }
    public float[] Beta { get; }
    public float Epsilon { get; }
    public int Dimension => Gamma.Length;
}

public sealed class MultiHeadAttentionParameters
{
    public MultiHeadAttentionParameters(LinearWeights projection, LinearWeights outputProjection)
    {
        Projection = projection ?? throw new ArgumentNullException(nameof(projection));
        OutputProjection = outputProjection ?? throw new ArgumentNullException(nameof(outputProjection));
    }

    public LinearWeights Projection { get; }
    public LinearWeights OutputProjection { get; }
}

public sealed class FeedForwardParameters
{
    public FeedForwardParameters(LinearWeights upProjection, LinearWeights downProjection)
    {
        UpProjection = upProjection ?? throw new ArgumentNullException(nameof(upProjection));
        DownProjection = downProjection ?? throw new ArgumentNullException(nameof(downProjection));
    }

    public LinearWeights UpProjection { get; }
    public LinearWeights DownProjection { get; }
}

public sealed class TransformerBlockParameters
{
    public TransformerBlockParameters(
        MultiHeadAttentionParameters attention,
        FeedForwardParameters feedForward,
        LayerNormParameters layerNorm1,
        LayerNormParameters layerNorm2)
    {
        Attention = attention ?? throw new ArgumentNullException(nameof(attention));
        FeedForward = feedForward ?? throw new ArgumentNullException(nameof(feedForward));
        LayerNorm1 = layerNorm1 ?? throw new ArgumentNullException(nameof(layerNorm1));
        LayerNorm2 = layerNorm2 ?? throw new ArgumentNullException(nameof(layerNorm2));
    }

    public MultiHeadAttentionParameters Attention { get; }
    public FeedForwardParameters FeedForward { get; }
    public LayerNormParameters LayerNorm1 { get; }
    public LayerNormParameters LayerNorm2 { get; }
}

public sealed class Gpt2Parameters
{
    public Gpt2Parameters(
        float[,] tokenEmbeddings,
        float[,] positionalEmbeddings,
        IReadOnlyList<TransformerBlockParameters> blocks,
        LayerNormParameters finalLayerNorm)
    {
        ArgumentNullException.ThrowIfNull(tokenEmbeddings);
        ArgumentNullException.ThrowIfNull(positionalEmbeddings);
        ArgumentNullException.ThrowIfNull(blocks);

        if (blocks.Count == 0)
            throw new ArgumentException("At least one transformer block is required.", nameof(blocks));

        TokenEmbeddings = tokenEmbeddings;
        PositionalEmbeddings = positionalEmbeddings;
        Blocks = blocks;
        FinalLayerNorm = finalLayerNorm ?? throw new ArgumentNullException(nameof(finalLayerNorm));
    }

    public float[,] TokenEmbeddings { get; }
    public float[,] PositionalEmbeddings { get; }
    public IReadOnlyList<TransformerBlockParameters> Blocks { get; }
    public LayerNormParameters FinalLayerNorm { get; }
}

public sealed class Gpt2Model
{
    private const float NegativeInfinity = -1e9f;
    private const float SqrtTwoOverPi = 0.7978845608f; // sqrt(2 / pi)

    public Gpt2Model(Gpt2Config config)
    {
        Config = config ?? throw new ArgumentNullException(nameof(config));
    }

    public Gpt2Config Config { get; }

    public float[,] Forward(IReadOnlyList<int> inputTokens, Gpt2Parameters parameters)
    {
        ArgumentNullException.ThrowIfNull(inputTokens);
        ArgumentNullException.ThrowIfNull(parameters);

        if (inputTokens.Count == 0)
            throw new ArgumentException("Input must contain at least one token.", nameof(inputTokens));
        if (inputTokens.Count > Config.ContextSize)
            throw new InvalidOperationException($"Sequence length {inputTokens.Count} exceeds context size {Config.ContextSize}.");

        ValidateParameters(parameters);

        float[,] hidden = EmbedTokens(inputTokens, parameters.TokenEmbeddings, parameters.PositionalEmbeddings);
        foreach (TransformerBlockParameters block in parameters.Blocks)
        {
            hidden = TransformerBlock(hidden, block);
        }

        float[,] normalized = LayerNorm(hidden, parameters.FinalLayerNorm);
        return ProjectToVocab(normalized, parameters.TokenEmbeddings);
    }

    public IReadOnlyList<int> Generate(IList<int> promptTokens, Gpt2Parameters parameters, int tokensToGenerate)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);
        ArgumentNullException.ThrowIfNull(parameters);
        if (tokensToGenerate < 0)
            throw new ArgumentOutOfRangeException(nameof(tokensToGenerate));

        List<int> buffer = new(promptTokens);

        for (int step = 0; step < tokensToGenerate; step++)
        {
            if (buffer.Count == 0)
                throw new InvalidOperationException("Prompt must contain at least one token.");
            if (buffer.Count > Config.ContextSize)
                throw new InvalidOperationException("Prompt exceeds configured context size.");

            float[,] logits = Forward(buffer, parameters);
            int nextToken = ArgMaxLastRow(logits);
            buffer.Add(nextToken);
        }

        if (tokensToGenerate == 0)
            return Array.Empty<int>();

        return buffer.Skip(buffer.Count - tokensToGenerate).ToArray();
    }

    private void ValidateParameters(Gpt2Parameters parameters)
    {
        int vocabRows = parameters.TokenEmbeddings.GetLength(0);
        int embedSize = parameters.TokenEmbeddings.GetLength(1);
        if (vocabRows != Config.VocabularySize)
            throw new InvalidOperationException($"Token embedding table must have {Config.VocabularySize} rows but has {vocabRows}.");
        if (embedSize != Config.EmbeddingSize)
            throw new InvalidOperationException($"Embedding dimension mismatch. Expected {Config.EmbeddingSize} but got {embedSize}.");

        if (parameters.PositionalEmbeddings.GetLength(0) < Config.ContextSize)
            throw new InvalidOperationException("Positional embeddings do not cover the configured context size.");
        if (parameters.PositionalEmbeddings.GetLength(1) != Config.EmbeddingSize)
            throw new InvalidOperationException("Positional embedding dimension mismatch.");

        if (parameters.Blocks.Count != Config.LayerCount)
            throw new InvalidOperationException($"Model expects {Config.LayerCount} transformer blocks but received {parameters.Blocks.Count}.");

        foreach (TransformerBlockParameters block in parameters.Blocks)
        {
            ValidateBlock(block);
        }

        if (parameters.FinalLayerNorm.Dimension != Config.EmbeddingSize)
            throw new InvalidOperationException("Final layer norm dimension does not match embedding dimension.");
    }

    private void ValidateBlock(TransformerBlockParameters block)
    {
        EnsureLinearDims(block.Attention.Projection, Config.EmbeddingSize, Config.EmbeddingSize * 3);
        EnsureLinearDims(block.Attention.OutputProjection, Config.EmbeddingSize, Config.EmbeddingSize);
        EnsureLinearDims(block.FeedForward.UpProjection, Config.EmbeddingSize, Config.EmbeddingSize * 4);
        EnsureLinearDims(block.FeedForward.DownProjection, Config.EmbeddingSize * 4, Config.EmbeddingSize);

        if (block.LayerNorm1.Dimension != Config.EmbeddingSize || block.LayerNorm2.Dimension != Config.EmbeddingSize)
            throw new InvalidOperationException("LayerNorm dimensions must match embedding dimension.");
    }

    private static void EnsureLinearDims(LinearWeights weights, int expectedInput, int expectedOutput)
    {
        int inputDim = weights.Weights.GetLength(0);
        int outputDim = weights.Weights.GetLength(1);
        if (inputDim != expectedInput || outputDim != expectedOutput)
            throw new InvalidOperationException($"Linear layer dimensions are {inputDim}x{outputDim} but expected {expectedInput}x{expectedOutput}.");
    }

    private float[,] TransformerBlock(float[,] x, TransformerBlockParameters block)
    {
        float[,] norm1 = LayerNorm(x, block.LayerNorm1);
        float[,] attn = MultiHeadAttention(norm1, block.Attention);
        float[,] afterAttn = AddMatrices(x, attn);

        float[,] norm2 = LayerNorm(afterAttn, block.LayerNorm2);
        float[,] mlp = FeedForward(norm2, block.FeedForward);
        return AddMatrices(afterAttn, mlp);
    }

    private static float[,] FeedForward(float[,] input, FeedForwardParameters parameters)
    {
        float[,] up = Linear(input, parameters.UpProjection);
        float[,] activated = Gelu(up);
        return Linear(activated, parameters.DownProjection);
    }

    private float[,] MultiHeadAttention(float[,] input, MultiHeadAttentionParameters parameters)
    {
        float[,] qkv = Linear(input, parameters.Projection);
        int seqLen = input.GetLength(0);
        int embedDim = input.GetLength(1);
        int headDim = Config.HeadSize;
        int headCount = Config.HeadCount;

        float[,] q = SliceColumns(qkv, 0, embedDim);
        float[,] k = SliceColumns(qkv, embedDim, embedDim);
        float[,] v = SliceColumns(qkv, 2 * embedDim, embedDim);

        float[,,] qHeads = SplitHeads(q, headCount, headDim);
        float[,,] kHeads = SplitHeads(k, headCount, headDim);
        float[,,] vHeads = SplitHeads(v, headCount, headDim);
        float[,,] attended = new float[headCount, seqLen, headDim];

        float[,] mask = BuildCausalMask(seqLen);
        float scale = 1f / MathF.Sqrt(headDim);

        for (int head = 0; head < headCount; head++)
        {
            float[,] scores = new float[seqLen, seqLen];
            for (int row = 0; row < seqLen; row++)
                for (int col = 0; col < seqLen; col++)
                {
                    float dot = 0f;
                    for (int dim = 0; dim < headDim; dim++)
                        dot += qHeads[head, row, dim] * kHeads[head, col, dim];
                    scores[row, col] = dot * scale + mask[row, col];
                }

            float[,] probs = SoftmaxRows(scores);

            for (int row = 0; row < seqLen; row++)
                for (int dim = 0; dim < headDim; dim++)
                {
                    float sum = 0f;
                    for (int col = 0; col < seqLen; col++)
                        sum += probs[row, col] * vHeads[head, col, dim];
                    attended[head, row, dim] = sum;
                }
        }

        float[,] merged = MergeHeads(attended, headCount, headDim);
        return Linear(merged, parameters.OutputProjection);
    }

    private static float[,] EmbedTokens(IReadOnlyList<int> tokens, float[,] tokenEmbeddings, float[,] positionalEmbeddings)
    {
        int seqLen = tokens.Count;
        int embedDim = tokenEmbeddings.GetLength(1);
        float[,] result = new float[seqLen, embedDim];

        for (int position = 0; position < seqLen; position++)
        {
            int tokenId = tokens[position];
            if (tokenId < 0 || tokenId >= tokenEmbeddings.GetLength(0))
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Token id {tokenId} is outside the vocabulary range.");

            for (int dim = 0; dim < embedDim; dim++)
            {
                float value = tokenEmbeddings[tokenId, dim];
                value += positionalEmbeddings[position, dim];
                result[position, dim] = value;
            }
        }

        return result;
    }

    private static float[,] Linear(float[,] input, LinearWeights weights)
    {
        int rows = input.GetLength(0);
        int inputDim = input.GetLength(1);
        int weightRows = weights.Weights.GetLength(0);
        int outputDim = weights.Weights.GetLength(1);

        if (inputDim != weightRows)
            throw new InvalidOperationException("Input dimension does not match weight rows.");

        float[,] output = new float[rows, outputDim];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < outputDim; col++)
            {
                float sum = weights.Bias[col];
                for (int k = 0; k < inputDim; k++)
                    sum += input[row, k] * weights.Weights[k, col];
                output[row, col] = sum;
            }
        }

        return output;
    }

    private static float[,] Gelu(float[,] input)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);
        float[,] output = new float[rows, cols];

        for (int row = 0; row < rows; row++)
            for (int col = 0; col < cols; col++)
            {
                float x = input[row, col];
                float cubic = x * x * x;
                float tanhArg = SqrtTwoOverPi * (x + 0.044715f * cubic);
                float tanhValue = MathF.Tanh(tanhArg);
                output[row, col] = 0.5f * x * (1f + tanhValue);
            }

        return output;
    }

    private static float[,] LayerNorm(float[,] input, LayerNormParameters parameters)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);
        if (cols != parameters.Dimension)
            throw new InvalidOperationException("LayerNorm dimension mismatch.");

        float[,] output = new float[rows, cols];
        for (int row = 0; row < rows; row++)
        {
            float mean = 0f;
            for (int col = 0; col < cols; col++)
                mean += input[row, col];
            mean /= cols;

            float variance = 0f;
            for (int col = 0; col < cols; col++)
            {
                float diff = input[row, col] - mean;
                variance += diff * diff;
            }
            variance /= cols;
            float invStd = 1f / MathF.Sqrt(variance + parameters.Epsilon);

            for (int col = 0; col < cols; col++)
            {
                float normalized = (input[row, col] - mean) * invStd;
                output[row, col] = normalized * parameters.Gamma[col] + parameters.Beta[col];
            }
        }

        return output;
    }

    private static float[,] AddMatrices(float[,] left, float[,] right)
    {
        int rows = left.GetLength(0);
        int cols = left.GetLength(1);
        if (rows != right.GetLength(0) || cols != right.GetLength(1))
            throw new InvalidOperationException("Residual add requires tensors with identical shapes.");

        float[,] result = new float[rows, cols];
        for (int row = 0; row < rows; row++)
            for (int col = 0; col < cols; col++)
                result[row, col] = left[row, col] + right[row, col];
        return result;
    }

    private static float[,] SliceColumns(float[,] source, int startColumn, int length)
    {
        int rows = source.GetLength(0);
        int cols = source.GetLength(1);
        if (startColumn < 0 || startColumn + length > cols)
            throw new ArgumentOutOfRangeException(nameof(startColumn));

        float[,] slice = new float[rows, length];
        for (int row = 0; row < rows; row++)
            for (int col = 0; col < length; col++)
                slice[row, col] = source[row, startColumn + col];
        return slice;
    }

    private static float[,,] SplitHeads(float[,] source, int headCount, int headDim)
    {
        int seqLen = source.GetLength(0);
        float[,,] result = new float[headCount, seqLen, headDim];
        for (int head = 0; head < headCount; head++)
        {
            int columnOffset = head * headDim;
            for (int position = 0; position < seqLen; position++)
                for (int dim = 0; dim < headDim; dim++)
                    result[head, position, dim] = source[position, columnOffset + dim];
        }
        return result;
    }

    private static float[,] MergeHeads(float[,,] source, int headCount, int headDim)
    {
        int seqLen = source.GetLength(1);
        float[,] merged = new float[seqLen, headCount * headDim];
        for (int head = 0; head < headCount; head++)
        {
            int columnOffset = head * headDim;
            for (int position = 0; position < seqLen; position++)
                for (int dim = 0; dim < headDim; dim++)
                    merged[position, columnOffset + dim] = source[head, position, dim];
        }
        return merged;
    }

    private static float[,] SoftmaxRows(float[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        float[,] result = new float[rows, cols];

        for (int row = 0; row < rows; row++)
        {
            float max = float.NegativeInfinity;
            for (int col = 0; col < cols; col++)
                if (matrix[row, col] > max)
                    max = matrix[row, col];

            float sum = 0f;
            for (int col = 0; col < cols; col++)
            {
                float value = MathF.Exp(matrix[row, col] - max);
                result[row, col] = value;
                sum += value;
            }

            float invSum = 1f / sum;
            for (int col = 0; col < cols; col++)
                result[row, col] *= invSum;
        }

        return result;
    }

    private static float[,] BuildCausalMask(int sequenceLength)
    {
        float[,] mask = new float[sequenceLength, sequenceLength];
        for (int row = 0; row < sequenceLength; row++)
            for (int col = 0; col < sequenceLength; col++)
                mask[row, col] = col <= row ? 0f : NegativeInfinity;
        return mask;
    }

    private static float[,] ProjectToVocab(float[,] hidden, float[,] tokenEmbeddings)
    {
        int seqLen = hidden.GetLength(0);
        int embedDim = hidden.GetLength(1);
        int vocabSize = tokenEmbeddings.GetLength(0);

        float[,] logits = new float[seqLen, vocabSize];
        for (int row = 0; row < seqLen; row++)
        {
            for (int vocab = 0; vocab < vocabSize; vocab++)
            {
                float sum = 0f;
                for (int dim = 0; dim < embedDim; dim++)
                    sum += hidden[row, dim] * tokenEmbeddings[vocab, dim];
                logits[row, vocab] = sum;
            }
        }
        return logits;
    }

    private static int ArgMaxLastRow(float[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        int lastRow = rows - 1;
        int argMax = 0;
        float best = matrix[lastRow, 0];
        for (int col = 1; col < cols; col++)
        {
            float value = matrix[lastRow, col];
            if (value > best)
            {
                best = value;
                argMax = col;
            }
        }
        return argMax;
    }
}
