using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Gpt2Infere;

/// <summary>
/// Byte Pair Encoding implementation adapted from the original Python GPT-2 encoder.
/// </summary>
internal sealed class BpeEncoder: Gpt2Encoder
{
    private readonly Dictionary<string, int> _encoder;
    private readonly Dictionary<int, string> _decoder;
    private readonly Dictionary<byte, string> _byteEncoder;
    private readonly Dictionary<char, byte> _byteDecoder;
    private readonly Dictionary<(string First, string Second), int> _bpeRanks;
    private readonly Dictionary<string, string> _cache = new();
    private readonly Regex _tokenPattern;
    private readonly Encoding _utf8;

    private BpeEncoder(
        Dictionary<string, int> encoder,
        IReadOnlyList<(string First, string Second)> merges,
        string errors)
    {
        _encoder = encoder;
        _decoder = encoder.ToDictionary(static kvp => kvp.Value, static kvp => kvp.Key);
        (_byteEncoder, _byteDecoder) = BuildByteUnicodeLookups();
        _bpeRanks = merges
            .Select((pair, index) => (pair, index))
            .ToDictionary(static x => x.pair, static x => x.index);
        _tokenPattern = new Regex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            RegexOptions.Compiled);
        _utf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: errors != "replace");
    }

    public static BpeEncoder FromDirectory(string modelsDirectory, string errors = "replace")
    {
        string encoderPath = Path.Combine(modelsDirectory, "encoder.json");
        string mergesPath = Path.Combine(modelsDirectory, "vocab.bpe");

        Dictionary<string, int>? encoder = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(encoderPath));
        if (encoder == null)
        {
            throw new InvalidOperationException($"Failed to deserialize encoder from '{encoderPath}'.");
        }

        string[] mergeLines = File.ReadAllLines(mergesPath);
        List<(string, string)> merges = mergeLines
            .Skip(1) // skip version line
            .Where(static line => !string.IsNullOrWhiteSpace(line))
            .Select(static line => line.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            .Where(static parts => parts.Length == 2)
            .Select(static parts => (parts[0], parts[1]))
            .ToList();

        return new BpeEncoder(encoder, merges, errors);
    }

    internal override int[] Encode(string text)
    {
        List<int> tokens = new();
        foreach (Match match in _tokenPattern.Matches(text))
        {
            string token = match.Value;
            string encoded = EncodeUtf8(token);
            string[] bpeTokens = ApplyBpe(encoded).Split(' ', StringSplitOptions.RemoveEmptyEntries);
            foreach (string bpeToken in bpeTokens)
            {
                if (!_encoder.TryGetValue(bpeToken, out int tokenId))
                {
                    throw new InvalidOperationException($"Token '{bpeToken}' not present in vocabulary.");
                }

                tokens.Add(tokenId);
            }
        }

        return tokens.ToArray();
    }

    public string Decode(IEnumerable<int> tokens)
    {
        StringBuilder textBuilder = new();
        foreach (int token in tokens)
        {
            if (!_decoder.TryGetValue(token, out string? piece))
            {
                throw new InvalidOperationException($"Token id '{token}' not present in decoder.");
            }

            textBuilder.Append(piece);
        }

        List<byte> byteBuffer = new(textBuilder.Length);
        foreach (char c in textBuilder.ToString())
        {
            if (!_byteDecoder.TryGetValue(c, out byte value))
            {
                throw new InvalidOperationException($"Character '{c}' missing from byte decoder.");
            }
            byteBuffer.Add(value);
        }

        return _utf8.GetString(byteBuffer.ToArray());
    }

    private string ApplyBpe(string token)
    {
        if (_cache.TryGetValue(token, out string? cached))
        {
            return cached;
        }

        List<string> word = token.Select(static c => c.ToString()).ToList();
        HashSet<(string, string)> pairs = GetPairs(word);

        if (pairs.Count == 0)
        {
            _cache[token] = token;
            return token;
        }

        while (pairs.Count > 0)
        {
            (string, string) bigram = pairs
                .OrderBy(pair => _bpeRanks.TryGetValue(pair, out int value) ? value : int.MaxValue)
                .First();

            if (!_bpeRanks.ContainsKey(bigram))
            {
                break;
            }

            List<string> newWord = new();
            string first = bigram.Item1;
            string second = bigram.Item2;
            int i = 0;

            while (i < word.Count)
            {
                int j = word.FindIndex(i, s => s == first);
                if (j == -1)
                {
                    newWord.AddRange(word.GetRange(i, word.Count - i));
                    break;
                }

                if (j > i)
                {
                    newWord.AddRange(word.GetRange(i, j - i));
                    i = j;
                }

                if (i < word.Count - 1 && word[i] == first && word[i + 1] == second)
                {
                    newWord.Add(first + second);
                    i += 2;
                }
                else
                {
                    newWord.Add(word[i]);
                    i += 1;
                }
            }

            word = newWord;
            if (word.Count == 1)
            {
                break;
            }

            pairs = GetPairs(word);
        }

        string result = string.Join(' ', word);
        _cache[token] = result;
        return result;
    }

    private static HashSet<(string, string)> GetPairs(IReadOnlyList<string> word)
    {
        HashSet<(string, string)> pairs = new();
        if (word.Count < 2)
        {
            return pairs;
        }

        string prev = word[0];
        for (int i = 1; i < word.Count; i++)
        {
            string current = word[i];
            pairs.Add((prev, current));
            prev = current;
        }

        return pairs;
    }

    private static (Dictionary<byte, string> Encoder, Dictionary<char, byte> Decoder) BuildByteUnicodeLookups()
    {
        List<int> bs = new();
        bs.AddRange(Enumerable.Range('!', '~' - '!' + 1));
        bs.AddRange(Enumerable.Range('¡', '¬' - '¡' + 1));
        bs.AddRange(Enumerable.Range('®', 'ÿ' - '®' + 1));

        HashSet<int> existing = new(bs);
        List<int> cs = new(bs);
        int n = 0;

        for (int b = 0; b < 256; b++)
        {
            if (existing.Contains(b))
            {
                continue;
            }

            bs.Add(b);
            cs.Add(256 + n);
            existing.Add(b);
            n++;
        }

        Dictionary<byte, string> byteEncoder = new(bs.Count);
        Dictionary<char, byte> byteDecoder = new(bs.Count);
        for (int i = 0; i < bs.Count; i++)
        {
            byte b = (byte)bs[i];
            char c = (char)cs[i];
            byteEncoder[b] = c.ToString();
            byteDecoder[c] = b;
        }

        return (byteEncoder, byteDecoder);
    }

    private string EncodeUtf8(string token)
    {
        byte[] utf8Bytes = _utf8.GetBytes(token);
        StringBuilder encoded = new(token.Length);
        foreach (byte b in utf8Bytes)
        {
            encoded.Append(_byteEncoder[b]);
        }

        return encoded.ToString();
    }

    internal override string Decode(int[] tokenIds)
    {
        return Decode(tokenIds.AsEnumerable());
    }
}
