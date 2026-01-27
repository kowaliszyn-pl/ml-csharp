// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace Gpt2Tokenization;

internal class Program
{
    static void Main(string[] args)
    {
        const string Text =
@"bajt bajtowi bajtem bajty bajtów bajtach bajtami bajtowie
To jest przykładowy tekst do trenowania tokenizatora GPT-2.
Tekst jest w języku polskim i zawiera znaki diakrytyczne, takie jak ą, ć, ę, ł, ń, ó, ś, ź, ż.
Będziemy próbować dzielić go na końcówki fleksyjne: -ami, -em, -owi, -ą, -ę, -u, -y, -i, -ów, -ach, -ami, -owie, -e, -a, -o, -ska, -ski, -skie, -ego, -ej, -ym, -ych, -emu, -ej, -iem, -ią.
Będziemy obserwować, jak tokenizator radzi sobie z tymi końcówkami i czy potrafi je poprawnie identyfikować oraz jak podczas późniejszego trenowania modelu językowego wpływa to na jakość generowanego tekstu, a w szczególności jak kształtować się będą osadzenia dla rdzeni wyrazów bez końcówek fleksyjnych.

Test apostrofów: I'll be there. You're amazing! It's a beautiful day.

Podpisano
Roman Kowaliszyn
www.kowaliszyn.pl

Słowa:
królami królem królowi króla królowe królów królach królami królowie 
należny należnemu należnej należnym

😎
";
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Gpt2Tokenizer tokenizer = Gpt2Tokenizer.TrainFromText(Text, 6000, 600);

        int[] tokens = tokenizer.Encode(Text);
        string decodedTokens = string.Join("\n", tokens
            .Select(tokenId => new { tokenId, tokenString = tokenizer.Decode(tokenId) })
            .Select(t => $"[{t.tokenId}: \"{t.tokenString}\"]"));

        Console.WriteLine(decodedTokens);
        Console.WriteLine($"Token count: {tokens.Length}");
        Console.ReadLine();
    }
}
