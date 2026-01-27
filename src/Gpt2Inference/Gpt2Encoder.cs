// Neural Networks in C♯
// File name: Gpt2Encoder.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

/// <summary>
/// Byte Pair Encoding implementation adapted from the original Python GPT-2 encoder.
/// </summary>
internal sealed partial class Gpt2Encoder
{
    private readonly Dictionary<string, int> _encoder;
    private readonly Dictionary<int, string> _decoder;
    private readonly Dictionary<byte, string> _byteEncoder;
    private readonly Dictionary<char, byte> _byteDecoder;
    private readonly Dictionary<(string First, string Second), int> _bpeRanks;
    private readonly Dictionary<string, string> _cache = [];
    private readonly Regex _tokenPattern;
    private readonly Encoding _utf8;

    private Gpt2Encoder(
        Dictionary<string, int> encoder,
        IReadOnlyList<(string First, string Second)> merges,
        bool throwOnInvalidBytes)
    {
        _encoder = encoder;
        _decoder = encoder.ToDictionary(static kvp => kvp.Value, static kvp => kvp.Key);
        (_byteEncoder, _byteDecoder) = BuildByteUnicodeLookups();
        _bpeRanks = merges
            .Select((pair, index) => (pair, index))
            .ToDictionary(static x => x.pair, static x => x.index);
        _tokenPattern = TokenizationPattern();
        _utf8 = new UTF8Encoding(false, throwOnInvalidBytes);
    }

    public static Gpt2Encoder CreateDummy(Gpt2HParams hParams)
    {
        int vocabSize = hParams.VocabularySize;
        Dictionary<string, int> encoder = new(vocabSize);
        for (int i = 0; i < vocabSize; i++)
        {
            encoder[$"token_{i}"] = i;
        }
        List<(string, string)> merges = [];
        return new Gpt2Encoder(encoder, merges, throwOnInvalidBytes: false);
    }

    public static Gpt2Encoder FromDirectory(string modelDirectory, bool throwOnInvalidBytes = false)
    {
        string encoderPath = Path.Combine(modelDirectory, "encoder.json");
        string mergesPath = Path.Combine(modelDirectory, "vocab.bpe");

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

        return new Gpt2Encoder(encoder, merges, throwOnInvalidBytes);
    }

    public void SaveToDirectory(string modelDirectory)
    {
        string encoderPath = Path.Combine(modelDirectory, "encoder.json");
        string mergesPath = Path.Combine(modelDirectory, "vocab.bpe");
        string encoderJson = JsonSerializer.Serialize(_encoder, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(encoderPath, encoderJson);
        List<string> mergeLines = ["#version: 0.2"];
        foreach (var pair in _bpeRanks.OrderBy(kvp => kvp.Value).Select(kvp => kvp.Key))
        {
            mergeLines.Add($"{pair.First} {pair.Second}");
        }
        File.WriteAllLines(mergesPath, mergeLines);
    }

    public static Gpt2Encoder CreateCustom(
        Dictionary<string, int> encoder,
        IReadOnlyList<(string First, string Second)> merges,
        bool throwOnInvalidBytes = false)
    {
        return new Gpt2Encoder(encoder, merges, throwOnInvalidBytes);
    }

    public static Gpt2Encoder TrainFromText(
        string text,
        int vocabSize,
        int numMerges,
        bool throwOnInvalidBytes = false)
    {
        // Build base vocabulary from byte encoder (256 base tokens)
        (Dictionary<byte, string> byteEncoder, _) = BuildByteUnicodeLookups();
        Dictionary<string, int> encoder = [];
        int tokenId = 0;
        foreach (string byteToken in byteEncoder.Values)
        {
            encoder[byteToken] = tokenId++;
        }

        // Use TokenizationPattern to split text into tokens
        Regex tokenPattern = TokenizationPattern();
        UTF8Encoding utf8 = new(false, throwOnInvalidBytes);
        List<string> tokens = [];
        foreach (Match match in tokenPattern.Matches(text))
        {
            byte[] tokenBytes = utf8.GetBytes(match.Value);
            tokens.AddRange(tokenBytes.Select(b => byteEncoder[b]));
        }

        // Group tokens into words (each match is a "word" for BPE training)
        List<List<string>> words = tokenPattern.Matches(text)
            .Select(m =>
            {
                byte[] tokenBytes = utf8.GetBytes(m.Value);
                return tokenBytes.Select(b => byteEncoder[b]).ToList();
            })
            .ToList();

        List<(string First, string Second)> merges = [];

        for (int merge = 0; merge < numMerges && encoder.Count < vocabSize; merge++)
        {
            // Count all adjacent pairs across all words
            Dictionary<(string, string), int> pairCounts = [];
            foreach (List<string> word in words)
            {
                for (int i = 0; i < word.Count - 1; i++)
                {
                    var pair = (word[i], word[i + 1]);
                    pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + 1;
                }
            }

            if (pairCounts.Count == 0)
            {
                break;
            }

            // Find most frequent pair
            var bestPair = pairCounts.MaxBy(static kvp => kvp.Value).Key;

            // Create new merged token
            string newToken = bestPair.Item1 + bestPair.Item2;
            if (!encoder.ContainsKey(newToken))
            {
                encoder[newToken] = tokenId++;
            }

            merges.Add((bestPair.Item1, bestPair.Item2));

            // Apply merge to all words
            for (int w = 0; w < words.Count; w++)
            {
                words[w] = ApplyMerge(words[w], bestPair.Item1, bestPair.Item2, newToken);
            }
        }

        return new Gpt2Encoder(encoder, merges, throwOnInvalidBytes);
    }

    private static List<string> ApplyMerge(List<string> word, string first, string second, string merged)
    {
        List<string> newWord = [];
        int i = 0;

        while (i < word.Count)
        {
            if (i < word.Count - 1 && word[i] == first && word[i + 1] == second)
            {
                newWord.Add(merged);
                i += 2;
            }
            else
            {
                newWord.Add(word[i]);
                i++;
            }
        }

        return newWord;
    }

    public int[] Encode(string text)
    {
        List<int> tokens = [];
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

    public string Decode(params IEnumerable<int> tokens)
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

            List<string> newWord = [];
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
        HashSet<(string, string)> pairs = [];
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
        List<int> bs =
        [
            .. Enumerable.Range('!', '~' - '!' + 1),
            .. Enumerable.Range('¡', '¬' - '¡' + 1),
            .. Enumerable.Range('®', 'ÿ' - '®' + 1),
        ];

        HashSet<int> existing = [.. bs];
        List<int> cs = [.. bs];
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

    [GeneratedRegex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", RegexOptions.Compiled)]
    private static partial Regex TokenizationPattern();
}
