// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/jaymody/picoGPT (fork https://github.com/kowaliszyn-pl/pico-gpt-2)
// Also, part of the code also copied from https://github.com/lofcz/gpt2sharp (fork https://github.com/kowaliszyn-pl/sharp-gpt-2)

using System.Text.Json.Serialization;

namespace Gpt2Inference;

public class Gpt2HParams
{
    [JsonPropertyName("n_ctx")]
    public int ContextSize { get; set; } = 1024;

    [JsonPropertyName("n_head")]
    public int HeadCount { get; set; } = 12;

    [JsonPropertyName("n_vocab")]
    public int VocabularySize { get; set; } = 50257;

    [JsonPropertyName("n_embd")]
    public int EmbeddingSize { get; set; } = 768;

    [JsonPropertyName("n_layer")]
    public int LayerCount { get; set; } = 12;
    
    public int HeadSize 
        => HeadCount != 0 ? EmbeddingSize / HeadCount : 0; // = 64

    public static Gpt2HParams FromDirectory(string modelDirectory)
    {
        string path = Path.Combine(modelDirectory, "hparams.json");
        string json = File.ReadAllText(path);
        return System.Text.Json.JsonSerializer.Deserialize<Gpt2HParams>(json) 
            ?? throw new InvalidOperationException("Failed to deserialize hparams.json");
    }
}