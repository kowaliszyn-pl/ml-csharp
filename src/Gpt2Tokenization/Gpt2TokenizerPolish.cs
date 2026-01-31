// Neural Networks in C♯
// File name: Gpt2TokenizerPolish.cs
// www.kowaliszyn.pl, 2025 - 2026

using Gpt2Inference;

namespace Gpt2Tokenization;

internal class Gpt2TokenizerPolish(
    Dictionary<string, int> encoder,
    IReadOnlyList<(string First, string Second)> merges
) : Gpt2Tokenizer(encoder, merges)
{
    //[GeneratedRegex(@"(?<! )\b(ami|em|owi|ą|ę|u|y|i|ów|ach|owie|e|a|o|ska|ski|skie|nego|nej|nym|nych|emu|ej|iem|ią)\b| ?\p{L}+|[\+\-\*/]", RegexOptions.Compiled)]
    //[GeneratedRegex(@"owie| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.Compiled)]
    //private static partial Regex TestTokenizationPattern();
}
