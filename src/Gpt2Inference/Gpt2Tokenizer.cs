// Neural Networks in C♯
// File name: Gpt2Tokenizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Gpt2Inference;

/// <summary>
/// Byte Pair Encoding implementation adapted from the original Python GPT-2 encoder.
/// </summary>
public partial class Gpt2Tokenizer
{
    private const char SingleSpace = ' ';

    private readonly Dictionary<string, int> _textToTokenId; // If number of tokens <= 65,535 use ushort
    private readonly Dictionary<int, string> _tokenIdToText;
    private readonly Dictionary<byte, char> _byteToChar;
    private readonly Dictionary<char, byte> _charToByte;
    private readonly Dictionary<(string First, string Second), int> _pairRanks;
    private readonly Dictionary<string, string> _cache = [];
    private readonly Regex _wordPattern;
    private readonly UTF8Encoding _utf8Encoding;

    public Gpt2Tokenizer(
        Dictionary<string, int> textToTokenId,
        IReadOnlyList<(string First, string Second)> pairs,
        bool throwOnInvalidBytes = true
    )
    {
        _textToTokenId = textToTokenId;
        _tokenIdToText = textToTokenId.ToDictionary(static kvp => kvp.Value, static kvp => kvp.Key);
        (_byteToChar, _charToByte) = CreateByteCharMapping();
        _pairRanks = pairs
            .Select((pair, index) => (pair, index))
            .ToDictionary(static x => x.pair, static x => x.index);
        _wordPattern = GetWordPatternRegex();
        _utf8Encoding = new(false, throwOnInvalidBytes);
    }

    public static Gpt2Tokenizer CreateDummy(Gpt2HParams hParams)
    {
        int vocabSize = hParams.VocabularySize;
        Dictionary<string, int> encoder = new(vocabSize);
        for (int i = 0; i < vocabSize; i++)
        {
            encoder[$"token_{i}"] = i;
        }
        List<(string, string)> merges = [];
        return new Gpt2Tokenizer(encoder, merges, throwOnInvalidBytes: false);
    }

    public static Gpt2Tokenizer FromDirectory(string modelDirectory, bool throwOnInvalidBytes = false)
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
            .Select(static line => line.Split(SingleSpace, StringSplitOptions.RemoveEmptyEntries))
            .Where(static parts => parts.Length == 2)
            .Select(static parts => (parts[0], parts[1]))
            .ToList();

        return new Gpt2Tokenizer(encoder, merges, throwOnInvalidBytes);
    }

    public void SaveToDirectory(string modelDirectory)
    {
        string encoderPath = Path.Combine(modelDirectory, "encoder.json");
        string mergesPath = Path.Combine(modelDirectory, "vocab.bpe");
        string encoderJson = JsonSerializer.Serialize(_textToTokenId, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(encoderPath, encoderJson);
        List<string> mergeLines = ["#version: 0.2"];
        foreach ((string First, string Second) pair in _pairRanks.OrderBy(kvp => kvp.Value).Select(kvp => kvp.Key))
        {
            mergeLines.Add($"{pair.First} {pair.Second}");
        }
        File.WriteAllLines(mergesPath, mergeLines);
    }

    public static Gpt2Tokenizer CreateCustom(
        Dictionary<string, int> encoder,
        IReadOnlyList<(string First, string Second)> merges,
        bool throwOnInvalidBytes = false) => new(encoder, merges, throwOnInvalidBytes);

    public static Gpt2Tokenizer TrainFromText(
        string text,
        int vocabSize,
        int numMerges,
        bool throwOnInvalidBytes = false)
    {
        // Build base vocabulary from byte encoder (256 base tokens)
        (Dictionary<byte, char> byteEncoder, _) = CreateByteCharMapping();
        Dictionary<string, int> encoder = [];
        int tokenId = 0;
        foreach (char byteToken in byteEncoder.Values)
        {
            encoder[byteToken.ToString()] = tokenId++;
        }

        // Use TestTokenizationPattern to split text into tokens
        Regex tokenPattern = GetWordPatternRegex();
        UTF8Encoding utf8 = new(false, throwOnInvalidBytes);
        //List<string> tokens = [];
        //foreach (Match match in tokenPattern.Matches(text))
        //{
        //    byte[] tokenBytes = utf8.GetBytes(match.Value);
        //    tokens.AddRange(tokenBytes.Select(b => byteEncoder[b]));
        //}

        // Group tokens into words (each match is a "word" for BPE training)
        List<List<string>> words = tokenPattern.Matches(text)
            .Select(m =>
            {
                byte[] tokenBytes = utf8.GetBytes(m.Value);
                return tokenBytes.Select(b => byteEncoder[b].ToString())
                    .ToList();
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
                    (string, string) pair = (word[i], word[i + 1]);
                    pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + 1;
                }
            }

            if (pairCounts.Count == 0)
            {
                break;
            }

            // Find most frequent pair
            (string, string) bestPair = pairCounts.MaxBy(static kvp => kvp.Value).Key;

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

        return new Gpt2Tokenizer(encoder, merges, throwOnInvalidBytes);
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

    /// <summary>
    /// Encodes the specified text into an array of token identifiers using the current vocabulary.
    /// </summary>
    /// <param name="text">The input text to encode. Cannot be null.</param>
    /// <returns>An array of integers representing the token identifiers corresponding to the encoded text. The array will be
    /// empty if the input text contains no tokens.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the text contains a token that is not present in the vocabulary.</exception>
    public int[] Encode(string text)
    {
        List<int> tokenIds = [];

        IEnumerable<string> words = GetWords(text);

        foreach (string word in words)
        {
            // Encode word to custom UTF-8-based byte representation. Word " is" will be encoded to "Ġis".
            string encodedWord = EncodeUtf8(word);

            // Word "Poland" will be split to tokens "Pol", "and" and returned as "Pol and"
            string tokenTextsAsString = GetTokenTexts(encodedWord);

            // Split token texts by space. Word "Pol and" will be split to token texts: ["Pol", "and"]
            string[] tokenTexts = tokenTextsAsString.Split(SingleSpace, StringSplitOptions.RemoveEmptyEntries);
            foreach (string tokenText in tokenTexts)
            {
                if (!_textToTokenId.TryGetValue(tokenText, out int tokenId))
                {
                    throw new InvalidOperationException($"Token '{tokenText}' not present in vocabulary.");
                }

                tokenIds.Add(tokenId);
            }
        }

        return [.. tokenIds];
    }

    protected virtual IEnumerable<string> GetWords(string text)
        // Split text into words using the word pattern
        => _wordPattern.Matches(text).Select(m => m.Value);

    /// <summary>
    /// Encodes the specified string into a custom UTF-8-based encoded representation.
    /// </summary>
    /// <param name="word">The string to encode. Cannot be null.</param>
    /// <returns>A string containing the encoded representation of the input word.</returns>
    private string EncodeUtf8(string word)
    {
        byte[] utf8Bytes = _utf8Encoding.GetBytes(word);
        StringBuilder encoded = new(word.Length);
        foreach (byte utf8Byte in utf8Bytes)
        {
            encoded.Append(_byteToChar[utf8Byte]);
        }

        return encoded.ToString();
    }

    private string GetTokenTexts(string word)
    {
        // Check cache first to avoid redundant computations for the same word
        if (_cache.TryGetValue(word, out string? cached))
        {
            return cached;
        }

        // Split word into list of characters (initially each character is a separate token)
        List<string> wordParts = [.. word.Select(static c => c.ToString())];

        // Get all adjacent pairs in the current word representation
        HashSet<(string, string)> wordPairs = GetPairs(wordParts);

        // Iteratively merge the most frequent pair until no more merges are possible
        while (wordPairs.Count > 0)
        {
            (string first, string second) = wordPairs
                .OrderBy(pair => _pairRanks.TryGetValue(pair, out int value) ? value : int.MaxValue)
                .First();

            if (!_pairRanks.ContainsKey((first, second)))
            {
                break;
            }

            List<string> mergedWordParts = [];
            int wordPartIndex = 0;

            while (wordPartIndex < wordParts.Count)
            {
                int firstPartFromPairIndex = wordParts.FindIndex(wordPartIndex, s => s == first);

                // We have the following 4 cases to consider when merging pairs:

                // 1. No more occurrences of the first part of the pair - we can add all remaining parts and break
                if (firstPartFromPairIndex == -1)
                {
                    mergedWordParts.AddRange(wordParts.GetRange(wordPartIndex, wordParts.Count - wordPartIndex));
                    break;
                }

                // 2. There are some parts before the first part of the pair - we can add them all before merging
                if (firstPartFromPairIndex > wordPartIndex)
                {
                    mergedWordParts.AddRange(wordParts.GetRange(wordPartIndex, firstPartFromPairIndex - wordPartIndex));
                    wordPartIndex = firstPartFromPairIndex;
                }

                // 3. We found the first part of the pair and it is followed by the second part - we can merge them and skip both parts
                if (wordPartIndex < wordParts.Count - 1 && wordParts[wordPartIndex] == first && wordParts[wordPartIndex + 1] == second)
                {
                    mergedWordParts.Add(first + second);
                    wordPartIndex += 2;
                }
                // 4. We found the first part of the pair but it is not followed by the second part - we can add the first part and continue searching for the next occurrence
                else
                {
                    mergedWordParts.Add(wordParts[wordPartIndex]);
                    wordPartIndex += 1;
                }
            }

            wordParts = mergedWordParts;
            if (wordParts.Count == 1)
            {
                break;
            }

            wordPairs = GetPairs(wordParts);
        }

        string result = string.Join(SingleSpace, wordParts);
        _cache[word] = result;
        return result;
    }

    public string Decode(params IEnumerable<int> tokens)
    {
        StringBuilder textBuilder = new();
        foreach (int token in tokens)
        {
            if (!_tokenIdToText.TryGetValue(token, out string? piece))
            {
                throw new InvalidOperationException($"Token id '{token}' not present in decoder.");
            }

            textBuilder.Append(piece);
        }

        List<byte> byteBuffer = new(textBuilder.Length);
        foreach (char c in textBuilder.ToString())
        {
            if (!_charToByte.TryGetValue(c, out byte value))
            {
                throw new InvalidOperationException($"Character '{c}' missing from byte decoder.");
            }
            byteBuffer.Add(value);
        }

        return _utf8Encoding.GetString(byteBuffer.ToArray());
    }

    private static HashSet<(string, string)> GetPairs(List<string> word)
    {
        HashSet<(string, string)> pairs = [];
        string? first = word.FirstOrDefault();

        if (first != null)
        {
            for (int i = 1; i < word.Count; i++)
            {
                string second = word[i];
                pairs.Add((first, second));
                first = second;
            }
        }

        return pairs;
    }

    private static (Dictionary<byte, char> Encoder, Dictionary<char, byte> Decoder) CreateByteCharMapping()
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

        Dictionary<byte, char> byteEncoder = new(bs.Count);
        Dictionary<char, byte> byteDecoder = new(bs.Count);
        for (int i = 0; i < bs.Count; i++)
        {
            byte b = (byte)bs[i];
            char c = (char)cs[i];
            byteEncoder[b] = c;
            byteDecoder[c] = b;
        }

        return (byteEncoder, byteDecoder);
    }

    [GeneratedRegex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", RegexOptions.Compiled)]
    private static partial Regex GetWordPatternRegex();
}
