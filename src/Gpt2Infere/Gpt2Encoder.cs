// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/kowaliszyn-pl/pico-gpt-2 (fork)
// Also, part of the code also copied from https://github.com/kowaliszyn-pl/sharp-gpt-2 (fork)

internal abstract class Gpt2Encoder
{
    internal abstract string Decode(int[] tokeIds);

    internal abstract int[] Encode(string prompt);
}

internal class DummyGpt2Encoder : Gpt2Encoder
{
    private readonly int _vocabularySize;

    public DummyGpt2Encoder(int vocabularySize)
    {
        _vocabularySize = vocabularySize;
    }

    internal override string Decode(int[] tokeIds)
    {
        // Dummy implementation: convert token IDs back to characters
        return new string(tokeIds.Select(id => (char)(id % 256)).ToArray());
    }
    
    internal override int[] Encode(string prompt)
    {
        // Dummy implementation: each character's ASCII value modulo vocabulary size
        return prompt.Select(c => (int)c % _vocabularySize).ToArray();
    }
}