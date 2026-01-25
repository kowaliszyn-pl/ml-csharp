// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/kowaliszyn-pl/pico-gpt-2

internal class Gpt2Params
{
    public float[,] TokenEmbeddings { get; internal set; }
    public float[,] PositionalEmbeddings { get; internal set; }
}