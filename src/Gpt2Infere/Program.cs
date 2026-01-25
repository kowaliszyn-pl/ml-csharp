// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/kowaliszyn-pl/pico-gpt-2

using System.Text;

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

    private static (Gpt2Encoder encoder, Gpt2HParams hParams, Gpt2Params modelParams) LoadEncoderHParamsAndParams(string modelSize, string modelsDir) => throw new NotImplementedException();

    /*
        def generate(inputs, params, n_head, n_tokens_to_generate):
            from tqdm import tqdm

            for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
                logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
                next_id = np.argmax(logits[-1])  # greedy sampling
                inputs.append(int(next_id))  # append prediction to input

            return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids
     */

    private static IEnumerable<int> Generate(int[] inputIds, Gpt2Params modelParams, int headCount, int nTokensToGenerate) => throw new NotImplementedException();

    /*
        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
     */
}