// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

using Gpt2Inference;

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

Test matematyki:
2 + 3 = 5
kropki dziesiętne: 234.78 + 876.22 = 1111.0
przecinki dziesiętne: 234,78 + 876,22 = 1111,0
E = mc^2
Całka od 0 do nieskończoności z e^(-x) dx = 1

Podpisano
Roman Kowaliszyn
www.kowaliszyn.pl

Słowa:
królami królem królowi króla królowe królów królach królami królowie królewska królewskie królewskim królewskich królewskości królewskością
należny należnemu należnej należnym

😎

Po japońskui:
今日も遠くを眺，涙を流す。 (Kyō mo tōku o nage, namida o nagasu.)

Po rosyjsku:
Сегодня я смотрю вдаль и плачу.

Po polsku:
Dziś znowu patrzę w dal i płaczę.

Po angielsku:
Today I look into the distance and cry.

Po arabsku:
اليوم أنظر إلى المسافة وأبكي.

Po chińsku:
今天我望向远方，哭泣。

Po hebrajsku:
היום אני מביט למרחק ובוכה.
";
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        /*
        Gpt2Tokenizer tokenizer = Gpt2Tokenizer.TrainFromText(Text, 600, 1000);

        int[] tokens = tokenizer.Encode(Text);
        string decodedTokens = string.Join("\n", tokens
            .Select(tokenId => new { tokenId, tokenString = tokenizer.Decode(tokenId) })
            .Select(t => $"[{t.tokenId}: \"{t.tokenString}\"]"));

        Console.WriteLine(decodedTokens);
        Console.WriteLine($"\nToken count: {tokens.Length}");

        string decodedText = tokenizer.Decode(tokens);
        Console.WriteLine("\nDecoded text:");
        Console.WriteLine(decodedText);

        Console.ReadLine();*/
    }
}
