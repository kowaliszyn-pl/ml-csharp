// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/kowaliszyn-pl/pico-gpt-2 (fork)
// Also, part of the code also copied from https://github.com/kowaliszyn-pl/sharp-gpt-2 (fork)

using System.Text;

using static NeuralNetworks.Core.ArrayExtensions;

internal class Program
{
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
        int nTokensToGenerate = args.Length > 1 && int.TryParse(args[1], out var n) ? n : 40;
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

        foreach(int outputId in Generate(inputIds, modelParams, hParams.HeadCount, nTokensToGenerate))
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
            float[] logits = Forward(inputs.ToArray(), modelParams, headCount);
            int nextId = logits.Argmax();
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

    private static float[] Forward(int[] inputIds, Gpt2Params modelParams, int headCount)
    {
        // X is [inputTokens, embeddingSize]
        float[,] X = EmbedTokens(inputIds, modelParams.TokenEmbeddings, modelParams.PositionalEmbeddings);

        for (int blockIndex = 0; blockIndex < modelParams.Blocks.Length; blockIndex++)
        {
            Gpt2Block block = modelParams.Blocks[blockIndex];
            X = TransformerBlockForward(X, block, headCount);
        }

        return null; // temp
    }

    /*
        def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
            # multi-head causal self attention
            x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

            # position-wise feed forward network
            x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

            return x
    */

    private static float[,] TransformerBlockForward(float[,] x, Gpt2Block block, int headCount) => throw new NotImplementedException();

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


    private static (Gpt2Encoder encoder, Gpt2HParams hParams, Gpt2Params modelParams) LoadEncoderHParamsAndParams(string modelSize, string modelsDir) => throw new NotImplementedException();

    /*
        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
     */
}