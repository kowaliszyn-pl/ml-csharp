// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/kowaliszyn-pl/pico-gpt-2 (fork)
// Also, part of the code also copied from https://github.com/kowaliszyn-pl/sharp-gpt-2 (fork)

using System.IO;
using System.Linq;
using System.Text.Json;

using Gpt2Infere;

using NeuralNetworks.Core;
using NeuralNetworks.Transformers.Gpt2;

using static NeuralNetworks.Core.ArrayExtensions;
using static NeuralNetworks.Core.RandomUtils;

internal class Program
{
    private const float NegativeInfinity = -1e10f;

    /*
        def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
            from utils import load_encoder_hparams_and_params

            # load encoder, hparams, and params from the released open-ai gpt-2 files
            encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

            # encode the input string using the BPE tokenizer
            input_ids = encoder.encode(prompt)

            # make sure we are not surpassing the max sequence length of our model
            assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

            # generate output ids
            output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

            # decode the ids back into a string
            output_text = encoder.decode(output_ids)

            return output_text
    */

    private static void Main(string[] args)
    {
        // Interpret the args as prompt, n_tokens_to_generate, model_size, models_dir
        string prompt = args.Length > 0 ? args[0] : "";
        int nTokensToGenerate = args.Length > 1 && int.TryParse(args[1], out var n) ? n : 10;
        string modelSize = args.Length > 2 ? args[2] : "124M";
        string modelsDir = args.Length > 3 ? args[3] : "models";


        Console.WriteLine($"Prompt: {prompt}");
        Console.WriteLine($"n_tokens_to_generate: {nTokensToGenerate}");
        Console.WriteLine($"model_size: {modelSize}");
        Console.WriteLine($"models_dir: {modelsDir}");

        // Prepare the model - load encoder, hparams, and params from the released open-ai gpt-2 files
        (Gpt2Encoder encoder, Gpt2HParams hParams, Gpt2Params modelParams) = LoadEncoderHParamsAndParams(modelSize, modelsDir);
        int[] inputIds = encoder.Encode(prompt);

        if (inputIds.Length + nTokensToGenerate >= hParams.ContextSize)
        {
            throw new ArgumentException("Input prompt is too long for the model's context size.");
        }

        // Print out the input token ids

        Console.WriteLine("Input token ids: " + string.Join(", ", inputIds));

        foreach (int outputId in Generate(inputIds, modelParams, hParams.HeadCount, nTokensToGenerate))
        {
            Console.Write(encoder.Decode(new int[] { outputId }));
        }

        Console.WriteLine("\nPress ENTER...");
        Console.ReadLine();
    }

    /*
        def generate(inputs, params, n_head, n_tokens_to_generate):
            from tqdm import tqdm

            for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
                logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
                next_id = np.argmax(logits[-1])  # greedy sampling
                inputs.append(int(next_id))  # append prediction to input

            return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids
     */

    private static IEnumerable<int> Generate(int[] inputIds, Gpt2Params modelParams, int headCount, int nTokensToGenerate)
    {
        List<int> inputs = new List<int>(inputIds);
        for (int i = 0; i < nTokensToGenerate; i++)
        {
            float[,] logits = Forward(inputs.ToArray(), modelParams, headCount);
            float[] lastTokenLogits = logits.GetRow(logits.GetLength(0) - 1);
            int nextId = lastTokenLogits.Argmax();
            inputs.Add(nextId);
            yield return nextId;
        }
    }

    /*
        def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
            # token + positional embeddings
            x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

            # forward pass through n_layer transformer blocks
            for block in blocks:
                x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

            # projection to vocab
            x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
            return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]
    */

    private static float[,] Forward(int[] inputIds, Gpt2Params modelParams, int headCount)
    {
        // X is [inputTokens, embeddingSize]
        float[,] X = EmbedTokens(inputIds, modelParams.TokenEmbeddings, modelParams.PositionalEmbeddings);

        for (int blockIndex = 0; blockIndex < modelParams.Blocks.Length; blockIndex++)
        {
            Gpt2Block block = modelParams.Blocks[blockIndex];
            X = TransformerBlockForward(X, block, headCount);
        }

        X = LayerNormForward(X, modelParams.FinalLayerNorm);

        // Project to vocab: [n_seq, n_embd] -> [n_seq, n_vocab]
        float[,] logitsMatrix = X.MultiplyDot(modelParams.TokenEmbeddings.Transpose());
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

        return result; // of size [inputTokens, embeddingSize]
    }

    /*
       def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
           # multi-head causal self attention
           x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

           # position-wise feed forward network
           x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

           return x
   */

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

    /*
        def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
            # project up
            a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

            # project back down
            x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

            return x
    */

    private static float[,] FeedForwardNetwork(float[,] x, Gpt2MultiLayerPerceptron mlp)
    {
        Gpt2LinearParams fullyConnected = mlp.FullyConnected;
        Gpt2LinearParams outputProjection = mlp.OutputProjection;

        // Project up: [n_seq, n_embd] -> [n_seq, 4*n_embd]
        // The size is 4 times larger in the hidden layer, because it allows the model to learn more complex representations. The 4 number is a design choice made by the authors of the Transformer architecture, and it has been found to work well in practice.
        float[,] a = LinearForward(x, fullyConnected);

        // Apply GELU activation
        a = a.Gelu();

        // Project back down: [n_seq, 4*n_embd] -> [n_seq, n_embd]
        float[,] output = LinearForward(a, outputProjection);
        return output;
    }

    /*
        def layer_norm(x, g, b, eps: float = 1e-5):
            mean = np.mean(x, axis=-1, keepdims=True)
            variance = np.var(x, axis=-1, keepdims=True)
            x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
            return g * x + b  # scale and offset with gamma/beta params
    */

    private static float[,] LayerNormForward(float[,] x, Gpt2LayerNormParams layerNorm)
    {
        float[] gamma = layerNorm.Gamma;
        float[] beta = layerNorm.Beta;

        if (gamma.Length != beta.Length)
            throw new ArgumentException("Gamma and beta must have the same length.");

        x.Standardize(); // TODO: check if the standardization is done over the correct (last) axis. also our Standardize implementation is a little different (no epsilon)

        return x.MultiplyElementwise(gamma).AddRow(beta); // TODO: check if the broadcasting is done correctly; AddRow or AddColumn?
    }

    /*
        def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
            # qkv projection
            x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

            # split into qkv
            qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

            # split into heads
            qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

            # causal mask to hide future inputs from being attended to
            causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

            # perform attention over each head
            out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

            # merge heads
            x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

            # out projection
            x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

            return x
    */

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
            // MergeHead(outHeads, headIndex, attn);

            //int nSeq = attn.GetLength(0);
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

    /*
        def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
            return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v
    */

    private static float[,] Attention(float[,] qh, float[,] kh, float[,] vh, float[,] causalMask)
    {
        int dK = qh.GetLength(1); // dK is the dimension of the key vectors, for GPT-2 124M it is 64
        float scale = 1f / (float)Math.Sqrt(dK);
        // q @ k.T
        float[,] qkT = qh.MultiplyDot(kh.Transpose());
        // Scale
        qkT = qkT.Multiply(scale);
        // Add causal mask
        qkT = qkT.Add(causalMask);
        // Softmax
        float[,] attnWeights = qkT.SoftmaxLogSumExp();
        // attnWeights @ v
        float[,] output = attnWeights.MultiplyDot(vh);
        return output;
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
            Gpt2HParams hParams = new Gpt2HParams
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
            SeededRandom seededRandom = new SeededRandom(42);

            Gpt2Params modelParams = new Gpt2Params
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
    {
        return BpeEncoder.FromDirectory(modelDirectory);


        //string encoderJson = Path.Combine(modelDirectory, "encoder.json");
        //string vocabBpe = Path.Combine(modelDirectory, "vocab.bpe");
        //return new BpeGpt2Encoder(encoderJson, vocabBpe);
    }
}