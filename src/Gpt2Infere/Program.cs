// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/kowaliszyn-pl/pico-gpt-2 (fork)
// Also, part of the code also copied from https://github.com/kowaliszyn-pl/sharp-gpt-2 (fork)

using System.Text.Json;

using Gpt2Infere;

using NeuralNetworks.Core;
using NeuralNetworks.Transformers.Gpt2;

using static NeuralNetworks.Core.ArrayExtensions;
using static NeuralNetworks.Core.RandomUtils;

internal class Program
{
    private const float NegativeInfinity = -1e10f;
    private const string ModelSize = "124M";
    private const string ModelsDir = "..\\..\\..\\..\\..\\data\\GPT-2\\";
    private const int NumTokensToGenerate = 20;

    private static void Main(string[] args)
    {
        Console.WriteLine($"NumTokensToGenerate: {NumTokensToGenerate}");
        Console.WriteLine($"ModelSize: {ModelSize}");
        Console.WriteLine($"ModelsDir: {ModelsDir}");

        // Prepare the model - load encoder, hparams, and params from the released open-ai gpt-2 files
        (Gpt2Encoder encoder, Gpt2HParams hParams, Gpt2Params modelParams) = LoadEncoderHParamsAndParams(ModelSize, ModelsDir);

        while (true)
        {
            Console.Write("Enter prompt: ");
            string? prompt = Console.ReadLine();

            if (string.IsNullOrEmpty(prompt))
                break;

            int[] inputIds = encoder.Encode(prompt);

            if (inputIds.Length + NumTokensToGenerate >= hParams.ContextSize)
            {
                throw new ArgumentException("Input prompt is too long for the model's context size.");
            }

            // Console.WriteLine("Input token ids: " + string.Join(", ", inputIds));

            foreach (int outputId in Generate(inputIds, modelParams, hParams.HeadCount, NumTokensToGenerate))
            {
                // Console.Write($" [{outputId}] ");
                Console.Write(encoder.Decode([outputId]));
            }
            Console.WriteLine();
            Console.WriteLine();
        }
        Console.WriteLine("\nPress ENTER...");
        Console.ReadLine();
    }

    private static IEnumerable<int> Generate(int[] inputIds, Gpt2Params modelParams, int headCount, int nTokensToGenerate)
    {
        List<int> inputs = [.. inputIds];
        for (int i = 0; i < nTokensToGenerate; i++)
        {
            float[,] logits = Forward(inputs.ToArray(), modelParams, headCount);
            float[] lastTokenLogits = logits.GetRow(logits.GetLength(0) - 1);
            int nextId = lastTokenLogits.Argmax();
            inputs.Add(nextId);
            yield return nextId;
        }
    }

    private static float[,] Forward(int[] inputIds, Gpt2Params modelParams, int headCount)
    {
        // X is [inputTokens, embeddingSize]
        float[,] x = EmbedTokens(inputIds, modelParams.TokenEmbeddings, modelParams.PositionalEmbeddings);

        for (int blockIndex = 0; blockIndex < modelParams.Blocks.Length; blockIndex++)
        {
            Gpt2Block block = modelParams.Blocks[blockIndex];
            x = TransformerBlockForward(x, block, headCount);
        }

        x = LayerNormForward(x, modelParams.FinalLayerNorm);

        // Project to vocab: [n_seq, n_embd] -> [n_seq, n_vocab]
        float[,] logitsMatrix = x.MultiplyDot(modelParams.TokenEmbeddings.Transpose());
        return logitsMatrix;
    }

    private static float[,] EmbedTokens(int[] inputTokenIds, float[,] tokenEmbeddings, float[,] positionalEmbeddings)
    {
        // tokenEmbeddings are of size [vocab_size, embedding_size],
        // where embedding_size is a size of the model embeddings (for GPT-2 124M it is 768)
        // and vocab_size is the size of the vocabulary (for GPT-2 124M it is 50257)

        // positionalEmbeddings are of size [context_size, embedding_size],
        // where context_size is the maximum context size of the model (for GPT-2 124M it is 1024)
        // where embedding_size is a size of the model embeddings (for GPT-2 124M it is 768)

        int inputTokens = inputTokenIds.Length;
        int embeddingSize = tokenEmbeddings.GetLength(1);
        float[,] result = new float[inputTokens, embeddingSize];

        for (int positionInInputSequence = 0; positionInInputSequence < inputTokens; positionInInputSequence++)
        {
            int tokenId = inputTokenIds[positionInInputSequence];
            if (tokenId < 0 || tokenId >= tokenEmbeddings.GetLength(0))
                throw new ArgumentOutOfRangeException(nameof(inputTokenIds), $"Token id {tokenId} is outside the vocabulary range.");

            // The purpose of this loop is to add token embeddings (for e given token) and positional embeddings (for a given position in the input sequence)
            for (int embeddingIndex = 0; embeddingIndex < embeddingSize; embeddingIndex++)
            {
                // For each position in the input sequence, we get the token embedding and add the positional embedding
                // embeddingIndex goes from 0 to 767 (for GPT-2 124M)
                float value = tokenEmbeddings[tokenId, embeddingIndex];
                value += positionalEmbeddings[positionInInputSequence, embeddingIndex];
                result[positionInInputSequence, embeddingIndex] = value;
            }
        }

        return result;
    }

    private static float[,] TransformerBlockForward(float[,] x, Gpt2Block block, int headCount)
    {
        // Multi-head causal self attention
        float[,] normalizedXForAttention = LayerNormForward(x, block.LayerNorm1);
        float[,] attentionOutput = MultiHeadAttention(normalizedXForAttention, block.Attention, headCount);
        x = x.Add(attentionOutput);

        // Position-wise feed forward network
        float[,] normalizedXForFeedForward = LayerNormForward(x, block.LayerNorm2);
        float[,] feedForwardOutput = FeedForwardNetwork(normalizedXForFeedForward, block.MultiLayerPerceptron);
        x = x.Add(feedForwardOutput);
        return x;
    }

    private static float[,] FeedForwardNetwork(float[,] x, Gpt2MultiLayerPerceptron mlp)
    {
        Gpt2LinearParams fullyConnected = mlp.FullyConnected;
        Gpt2LinearParams outputProjection = mlp.OutputProjection;

        // Project up: [n_seq, n_embd] -> [n_seq, 4*n_embd]
        // The size is 4 times larger in the hidden layer, because it allows the model to learn more complex representations. The 4 number is a design choice made by the authors of the Transformer architecture, and it has been found to work well in practice.
        float[,] hiddenLayer = LinearForward(x, fullyConnected);

        // Apply GELU activation
        hiddenLayer = hiddenLayer.Gelu();

        // Project back down: [n_seq, 4*n_embd] -> [n_seq, n_embd]
        float[,] output = LinearForward(hiddenLayer, outputProjection);
        return output;
    }

    private static float[,] LayerNormForward(float[,] x, Gpt2LayerNormParams layerNorm)
    {
        float[] gamma = layerNorm.Gamma;
        float[] beta = layerNorm.Beta;

        if (gamma.Length != beta.Length)
            throw new ArgumentException("Gamma and beta must have the same length.");

        float[,] normalized = x.StandardizeByRows(); // TODO: check if the standardization is done over the correct (last) axis. also our Standardize implementation is a little different (no epsilon)

        float[,] res = normalized.MultiplyElementwise(gamma).AddRow(beta); // TODO: check if the broadcasting is done correctly; AddRow or AddColumn?
        return res;
    }

    private static float[,] MultiHeadAttention(float[,] x, Gpt2MultiHeadAttentionParams attention, int headCount)
    {
        Gpt2LinearParams Projection = attention.Projection;

        // [n_seq, n_embd] -> [n_seq, 3*n_embd]
        x = LinearForward(x, Projection);

        // Split into qkv: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
        (float[,] q, float[,] k, float[,] v) = SplitIntoQKV(x);

        // Split into heads: [n_seq, n_embd] -> [n_head, n_seq, headDim]
        // where headDim = n_embd / n_head
        // In GPT-2 124M, n_embd = 768, n_head = 12, so headDim = 64
        // Interpretation of n_heads: each head is a separate attention mechanism that can focus on different parts of the input sequence. For example, one head might focus on syntactic structure, while another might focus on semantic meaning.
        float[,,] qHeads = SplitHeads(q, headCount);
        float[,,] kHeads = SplitHeads(k, headCount);
        float[,,] vHeads = SplitHeads(v, headCount);

        // Causal mask: [n_seq, n_seq]
        int inputSequenceLength = x.GetLength(0);
        float[,] causalMask = BuildCausalMask(inputSequenceLength);

        // Attention for each head
        int headDim = qHeads.GetLength(2);
        float[,,] outHeads = new float[headCount, inputSequenceLength, headDim]; // headCount * headDim = embedding size
        for (int headIndex = 0; headIndex < headCount; headIndex++)
        {
            // headIndex goes from 0 to 11 (for GPT-2 124M)
            float[,] qh = GetHead(qHeads, headIndex);
            float[,] kh = GetHead(kHeads, headIndex);
            float[,] vh = GetHead(vHeads, headIndex);
            float[,] attn = Attention(qh, kh, vh, causalMask); // [n_seq, headDim]

            for (int i = 0; i < inputSequenceLength; i++)
                for (int j = 0; j < headDim; j++)
                    outHeads[headIndex, i, j] = attn[i, j];
        }

        // Merge heads: [n_head, n_seq, headDim] -> [n_seq, n_embd]
        float[,] mergedHeads = new float[inputSequenceLength, headCount * headDim];
        for (int i = 0; i < inputSequenceLength; i++)
        {
            for (int h = 0; h < headCount; h++)
            {
                for (int j = 0; j < headDim; j++)
                {
                    mergedHeads[i, h * headDim + j] = outHeads[h, i, j];
                }
            }
        }

        Gpt2LinearParams outputProjection = attention.OutputProjection;
        // Out projection: [n_seq, n_embd] -> [n_seq, n_embd]
        float[,] output = LinearForward(mergedHeads, outputProjection);
        return output;
    }

    private static float[,] Attention(float[,] queryHeadMatrix, float[,] keyHeadMatrix, float[,] valueHeadMatrix, float[,] causalMask)
    {
        // Use the per-head channel count (dk) so that later scaling reflects how many features shape each query-key comparison.
        int keyEmbeddingWidth = queryHeadMatrix.GetLength(1); 
        
        // Precompute the transformer scaling factor 1/sqrt(dk) to keep attention logits numerically well-behaved regardless of head width.
        float inverseSqrtKeyDim = 1f / (float)Math.Sqrt(keyEmbeddingWidth); 

        // Measure how strongly every query token relates to every key token via dot products between their head-specific embeddings.
        float[,] rawAttentionScores = queryHeadMatrix.MultiplyDot(keyHeadMatrix.Transpose()); 

        // Temper the magnitude of those dot products so softmax probabilities remain sharp but not saturated when dk grows.
        rawAttentionScores = rawAttentionScores.Multiply(inverseSqrtKeyDim); 

        // Inject -inf above the diagonal to forbid each position from peeking at future tokens while leaving allowed positions unchanged.
        rawAttentionScores = rawAttentionScores.Add(causalMask); 

        // Turn the masked logits into normalized weights per query row using the numerically stable softmax implementation.
        float[,] attentionProbabilities = rawAttentionScores.SoftmaxStable();

        // Combine value vectors using the attention weights so each query token receives a context-aware representation.
        // Aggregate value vectors with the attention weights so each query token receives a context-aware representation.
        float[,] contextualizedValues = attentionProbabilities.MultiplyDot(valueHeadMatrix); 

        // Emit the attention head output that will later be concatenated with other heads and projected.
        return contextualizedValues;
    }

    private static float[,] GetHead(float[,,] heads, int headIndex)
    {
        int nSeq = heads.GetLength(1);
        int headDim = heads.GetLength(2);
        float[,] result = new float[nSeq, headDim];
        for (int i = 0; i < nSeq; i++)
            for (int j = 0; j < headDim; j++)
                result[i, j] = heads[headIndex, i, j];
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

    /// <summary>
    /// Split into heads: [n_seq, n_embd] -> [n_head, n_seq, headDim]
    /// </summary>
    /// <param name="qkv"></param>
    /// <param name="headCount"></param>
    /// <returns></returns>
    private static float[,,] SplitHeads(float[,] qkv, int headCount)
    {
        int nSeq = qkv.GetLength(0);
        int nEmbd = qkv.GetLength(1);
        if (nEmbd % headCount != 0)
            throw new ArgumentException("Embedding size must be divisible by head count.");
        int headDim = nEmbd / headCount;
        float[,,] result = new float[headCount, nSeq, headDim];
        for (int i = 0; i < nSeq; i++)
        {
            for (int h = 0; h < headCount; h++)
            {
                for (int j = 0; j < headDim; j++)
                {
                    result[h, i, j] = qkv[i, h * headDim + j];
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Splits a combined query, key, and value matrix into separate matrices for each component.
    /// </summary>
    /// <remarks>The input array must have a column count that is a multiple of 3. The method assumes that the
    /// first third of columns correspond to queries, the second third to keys, and the final third to values, in that
    /// order. So, it splits block-wise (first all q, then all k, then all v).
    /// </remarks>
    /// <param name="x">A two-dimensional array of shape [n_seq, 3 * n_embd], where each row contains concatenated query, key, and value
    /// vectors.</param>
    /// <returns>A tuple containing three two-dimensional arrays: the query matrix, the key matrix, and the value matrix. Each
    /// has shape [n_seq, n_embd].</returns>
    private static (float[,] q, float[,] k, float[,] v) SplitIntoQKV(float[,] x)
    {
        int nSeq = x.GetLength(0);
        int nEmbd = x.GetLength(1) / 3;
        float[,] q = new float[nSeq, nEmbd];
        float[,] k = new float[nSeq, nEmbd];
        float[,] v = new float[nSeq, nEmbd];
        for (int i = 0; i < nSeq; i++)
        {
            for (int j = 0; j < nEmbd; j++)
            {
                q[i, j] = x[i, j];
                k[i, j] = x[i, j + nEmbd];
                v[i, j] = x[i, j + 2 * nEmbd];
            }
        }
        return (q, k, v);
    }

    /// <summary>
    /// Applies a linear transformation to the input tensor using the specified weights and bias parameters.
    /// </summary>
    /// <param name="x">The input tensor to transform, represented as a two-dimensional array of shape [n_seq, n_embd]. Each row
    /// corresponds to a sequence element, and each column to an embedding dimension.</param>
    /// <param name="linearParams">The linear transformation parameters, including the weight matrix and bias vector to apply to the input tensor.</param>
    /// <returns>A two-dimensional array containing the result of the linear transformation. The returned array has shape [n_seq,
    /// output_size], where output_size is determined by the weights in the linear parameters.</returns>
    /// <remarks>This function can be treated as a 1D convolution with kernel size 1. It performs a simple operation: result = x * W + b.</remarks>
    private static float[,] LinearForward(float[,] x, Gpt2LinearParams linearParams)
    {
        // x is [n_seq, n_embd]
        float[,] weights = linearParams.Weights;
        float[] bias = linearParams.Bias;
        float[,] result = x.MultiplyDot(weights).AddRow(bias);

        // result is [n_seq, output_size]
        return result;
    }

    private static (Gpt2Encoder encoder, Gpt2HParams hParams, Gpt2Params modelParams) LoadEncoderHParamsAndParams(string modelSize, string modelsDir)
    {
        bool createDummy = modelSize == "0";

        if (createDummy)
        {
            Gpt2HParams hParams = new()
            {
                ContextSize = 1024,
                HeadCount = 12,
                VocabularySize = 50257,
                EmbeddingSize = 768,
                LayerCount = 12,
                //HeadSize = 64
            };

            Gpt2Encoder encoder = new DummyGpt2Encoder(hParams.VocabularySize);

            //Gpt2Params modelParams = new Gpt2Params
            //{
            //    TokenEmbeddings = new float[50257, 768],
            //    PositionalEmbeddings = new float[1024, 768],
            //    Blocks = new Gpt2Block[12],
            //    FinalLayerNorm = new Gpt2LayerNormParams
            //    {
            //        Gamma = new float[768],
            //        Beta = new float[768]
            //    }
            //};
            SeededRandom seededRandom = new(42);

            Gpt2Params modelParams = new()
            {
                TokenEmbeddings = CreateRandomNormal(50257, 768, seededRandom),
                PositionalEmbeddings = CreateRandomNormal(1024, 768, seededRandom),
                Blocks = Enumerable
                    .Range(0, 12)
                    .Select(_ => new Gpt2Block()
                    {
                        LayerNorm1 = new Gpt2LayerNormParams
                        {
                            Gamma = CreateRandomNormal(768, seededRandom),
                            Beta = CreateRandomNormal(768, seededRandom)
                        },
                        LayerNorm2 = new Gpt2LayerNormParams
                        {
                            Gamma = CreateRandomNormal(768, seededRandom),
                            Beta = CreateRandomNormal(768, seededRandom)
                        },
                        Attention = new Gpt2MultiHeadAttentionParams
                        {
                            Projection = new Gpt2LinearParams
                            {
                                Weights = CreateRandomNormal(768, 3 * 768, seededRandom),
                                Bias = CreateRandomNormal(3 * 768, seededRandom)
                            },
                            OutputProjection = new Gpt2LinearParams
                            {
                                Weights = CreateRandomNormal(768, 768, seededRandom),
                                Bias = CreateRandomNormal(768, seededRandom)
                            }
                        },
                        MultiLayerPerceptron = new Gpt2MultiLayerPerceptron
                        {
                            FullyConnected = new Gpt2LinearParams
                            {
                                Weights = CreateRandomNormal(768, 4 * 768, seededRandom),
                                Bias = CreateRandomNormal(4 * 768, seededRandom)
                            },
                            OutputProjection = new Gpt2LinearParams
                            {
                                Weights = CreateRandomNormal(4 * 768, 768, seededRandom),
                                Bias = CreateRandomNormal(768, seededRandom)
                            }
                        }
                    }
                    )
                    .ToArray(),
                FinalLayerNorm = new Gpt2LayerNormParams
                {
                    Gamma = CreateRandomNormal(768, seededRandom),
                    Beta = CreateRandomNormal(768, seededRandom)
                }
            };

            return (encoder, hParams, modelParams);
        }
        else
        {
            return LoadModel(modelSize, modelsDir);
        }
    }

    private static (Gpt2Encoder encoder, Gpt2HParams hParams, Gpt2Params modelParams) LoadModel(string modelSize, string modelsDir)
    {
        if (string.IsNullOrWhiteSpace(modelSize))
            throw new ArgumentException("Model size must be provided.", nameof(modelSize));

        if (string.IsNullOrWhiteSpace(modelsDir))
            throw new ArgumentException("Models directory must be provided.", nameof(modelsDir));

        string modelDirectory = ResolveModelDirectory(modelSize, modelsDir);
        string hparamsPath = Path.Combine(modelDirectory, "hparams.json");
        if (!File.Exists(hparamsPath))
            throw new FileNotFoundException($"Missing hparams.json for GPT-2 model '{modelSize}'.", hparamsPath);

        Gpt2HParams hParams = LoadHParamsFromFile(hparamsPath);
        string weightFilePath = ResolveWeightFile(modelDirectory);
        Gpt2Params modelParams = LoadParameters(weightFilePath, hParams);
        Gpt2Encoder encoder = CreateEncoder(modelDirectory, hParams);

        return (encoder, hParams, modelParams);
    }

    private static string ResolveModelDirectory(string modelSize, string modelsDir)
    {
        string resolvedModelsDir = Path.GetFullPath(modelsDir);
        if (!Directory.Exists(resolvedModelsDir))
            throw new DirectoryNotFoundException($"Models directory '{resolvedModelsDir}' does not exist.");

        string candidate = Path.Combine(resolvedModelsDir, modelSize);
        if (Directory.Exists(candidate))
            return candidate;

        DirectoryInfo resolvedInfo = new(resolvedModelsDir);
        if (string.Equals(resolvedInfo.Name, modelSize, StringComparison.OrdinalIgnoreCase))
            return resolvedModelsDir;

        throw new DirectoryNotFoundException($"Unable to locate GPT-2 files for '{modelSize}'. Expected to find them under '{candidate}'.");
    }

    private static Gpt2HParams LoadHParamsFromFile(string hparamsPath)
    {
        using FileStream stream = File.OpenRead(hparamsPath);
        using JsonDocument document = JsonDocument.Parse(stream);
        JsonElement root = document.RootElement;

        return new Gpt2HParams
        {
            ContextSize = ReadRequiredInt(root, "n_ctx", "n_positions"),
            HeadCount = ReadRequiredInt(root, "n_head"),
            VocabularySize = ReadRequiredInt(root, "n_vocab"),
            EmbeddingSize = ReadRequiredInt(root, "n_embd"),
            LayerCount = ReadRequiredInt(root, "n_layer")
        };
    }

    private static int ReadRequiredInt(JsonElement root, params string[] propertyNames)
    {
        foreach (string propertyName in propertyNames)
        {
            if (root.TryGetProperty(propertyName, out JsonElement element) && element.ValueKind == JsonValueKind.Number)
                return element.GetInt32();
        }

        throw new InvalidDataException($"The hparams.json file is missing required entries ({string.Join(", ", propertyNames)}).");
    }

    private static string ResolveWeightFile(string modelDirectory)
    {
        string directoryName = new DirectoryInfo(modelDirectory).Name;
        string[] preferredCandidates =
        {
            Path.Combine(modelDirectory, "gpt2-weights.bin"),
            Path.Combine(modelDirectory, "weights.bin"),
            Path.Combine(modelDirectory, $"{directoryName}.bin")
        };

        foreach (string candidate in preferredCandidates)
        {
            if (File.Exists(candidate))
                return candidate;
        }

        string[] binFiles = Directory.GetFiles(modelDirectory, "*.bin", SearchOption.TopDirectoryOnly);
        if (binFiles.Length == 1)
            return binFiles[0];

        if (binFiles.Length > 1)
        {
            string files = string.Join(", ", binFiles.Select(Path.GetFileName));
            throw new InvalidOperationException($"Multiple GPT-2 weight files were found in '{modelDirectory}': {files}. Delete the extras or pass a directory containing a single export.");
        }

        throw new FileNotFoundException($"No GPT-2 weight file (.bin) was found in '{modelDirectory}'. Run tools\\gpt2_export.py to create one.");
    }

    private static Gpt2Params LoadParameters(string weightFilePath, Gpt2HParams hParams)
    {
        Gpt2Config config = new(
            vocabularySize: hParams.VocabularySize,
            contextSize: hParams.ContextSize,
            embeddingSize: hParams.EmbeddingSize,
            headCount: hParams.HeadCount,
            layerCount: hParams.LayerCount);

        Gpt2Parameters raw = Gpt2ParameterLoader.LoadFromFile(weightFilePath, config);
        return ConvertParameters(raw);
    }

    private static Gpt2Params ConvertParameters(Gpt2Parameters parameters)
    {
        Gpt2Block[] blocks = parameters.Blocks.Select(ConvertBlock).ToArray();
        return new Gpt2Params
        {
            TokenEmbeddings = parameters.TokenEmbeddings,
            PositionalEmbeddings = parameters.PositionalEmbeddings,
            Blocks = blocks,
            FinalLayerNorm = ConvertLayerNorm(parameters.FinalLayerNorm)
        };
    }

    private static Gpt2Block ConvertBlock(TransformerBlockParameters block)
    {
        return new Gpt2Block
        {
            LayerNorm1 = ConvertLayerNorm(block.LayerNorm1),
            LayerNorm2 = ConvertLayerNorm(block.LayerNorm2),
            Attention = new Gpt2MultiHeadAttentionParams
            {
                Projection = ConvertLinear(block.Attention.Projection),
                OutputProjection = ConvertLinear(block.Attention.OutputProjection)
            },
            MultiLayerPerceptron = new Gpt2MultiLayerPerceptron
            {
                FullyConnected = ConvertLinear(block.FeedForward.UpProjection),
                OutputProjection = ConvertLinear(block.FeedForward.DownProjection)
            }
        };
    }

    private static Gpt2LayerNormParams ConvertLayerNorm(LayerNormParameters layerNorm)
    {
        return new Gpt2LayerNormParams
        {
            Gamma = layerNorm.Gamma,
            Beta = layerNorm.Beta
        };
    }

    private static Gpt2LinearParams ConvertLinear(LinearWeights linear)
    {
        return new Gpt2LinearParams
        {
            Weights = linear.Weights,
            Bias = linear.Bias
        };
    }

    private static Gpt2Encoder CreateEncoder(string modelDirectory, Gpt2HParams hParams)
        => BpeEncoder.FromDirectory(modelDirectory);
}