// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/kowaliszyn-pl/pico-gpt-2 (fork)
// Also, part of the code also copied from https://github.com/kowaliszyn-pl/sharp-gpt-2 (fork)

internal class Gpt2HParams
{
    public int ContextSize { get; internal set; }
    public int HeadCount { get; internal set; }
    public int VocabularySize { get; }
    public int EmbeddingSize { get; }
    public int LayerCount { get; }
    public int HeadSize => EmbeddingSize / HeadCount;
}