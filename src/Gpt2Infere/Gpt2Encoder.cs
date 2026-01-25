// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/kowaliszyn-pl/pico-gpt-2 (fork)
// Also, part of the code also copied from https://github.com/kowaliszyn-pl/sharp-gpt-2 (fork)

internal abstract class Gpt2Encoder
{
    internal abstract string Decode(int[] tokenIds);

    internal abstract int[] Encode(string prompt);
}

internal class DummyGpt2Encoder : Gpt2Encoder
{
    private readonly int _vocabularySize;

    public DummyGpt2Encoder(int vocabularySize)
    {
        _vocabularySize = vocabularySize;
    }

    internal override string Decode(int[] tokenIds)
    {
        // Dummy implementation: convert token IDs back to characters
        return new string(tokenIds.Select(id => (char)(id % 256)).ToArray());
    }
    
    internal override int[] Encode(string prompt)
    {
        // Dummy implementation: each character's ASCII value modulo vocabulary size
        return prompt.Select(c => (int)c % _vocabularySize).ToArray();
    }
}

internal class SimpleGpt2Encoder : Gpt2Encoder
{
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<int, string> _idToToken;
    public SimpleGpt2Encoder(Dictionary<string, int> tokenToId)
    {
        _tokenToId = tokenToId;
        _idToToken = tokenToId.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
    }
    internal override string Decode(int[] tokenIds)
    {
        var tokens = tokenIds.Select(id => _idToToken.ContainsKey(id) ? _idToToken[id] : "<unk>");
        return string.Join(" ", tokens);
    }
    
    internal override int[] Encode(string prompt)
    {
        var tokens = prompt.Split(' ');
        return tokens.Select(token => _tokenToId.ContainsKey(token) ? _tokenToId[token] : -1).ToArray();
    }
}


//internal class BpeGpt2Encoder: Gpt2Encoder
//{
//    private string _encoderJson;
//    private string _vocabBpe;

//    /*
//        def __init__(self, encoder, bpe_merges, errors="replace"):
//            self.encoder = encoder
//            self.decoder = {v: k for k, v in self.encoder.items()}
//            self.errors = errors  # how to handle errors in decoding
//            self.byte_encoder = bytes_to_unicode()
//            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
//            self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
//            self.cache = {}

//            # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
//            self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
//     * */

//    public BpeGpt2Encoder(string encoderJson, string vocabBpe)
//    {
//        _encoderJson = encoderJson;
//        _vocabBpe = vocabBpe;
//    }

//    /*
//        def bytes_to_unicode():
//            """
//            Returns list of utf-8 byte and a corresponding list of unicode strings.
//            The reversible bpe codes work on unicode strings.
//            This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
//            When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
//            This is a significant percentage of your normal, say, 32K bpe vocab.
//            To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
//            And avoids mapping to whitespace/control characters the bpe code barfs on.
//            """
//            bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
//            cs = bs[:]
//            n = 0
//            for b in range(2**8):
//                if b not in bs:
//                    bs.append(b)
//                    cs.append(2**8 + n)
//                    n += 1
//            cs = [chr(n) for n in cs]
//            return dict(zip(bs, cs))
//    */

//    /*
//         def get_pairs(word):
//            """Return set of symbol pairs in a word.
//            Word is represented as tuple of symbols (symbols being variable-length strings).
//            """
//            pairs = set()
//            prev_char = word[0]
//            for char in word[1:]:
//                pairs.add((prev_char, char))
//                prev_char = char
//            return pairs
//   */

//    /*
//        def bpe(self, token):
//            if token in self.cache:
//                return self.cache[token]
//            word = tuple(token)
//            pairs = get_pairs(word)

//            if not pairs:
//                return token

//            while True:
//                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
//                if bigram not in self.bpe_ranks:
//                    break
//                first, second = bigram
//                new_word = []
//                i = 0
//                while i < len(word):
//                    try:
//                        j = word.index(first, i)
//                        new_word.extend(word[i:j])
//                        i = j
//                    except:
//                        new_word.extend(word[i:])
//                        break

//                    if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
//                        new_word.append(first + second)
//                        i += 2
//                    else:
//                        new_word.append(word[i])
//                        i += 1
//                new_word = tuple(new_word)
//                word = new_word
//                if len(word) == 1:
//                    break
//                else:
//                    pairs = get_pairs(word)
//            word = " ".join(word)
//            self.cache[token] = word
//            return word
//     * *

//    /*
//        def decode(self, tokens):
//            text = "".join([self.decoder[token] for token in tokens])
//            text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
//            return text
//    */

//    internal override string Decode(int[] tokenIds)
//    {
//        string text = string.Concat(tokenIds.Select(id => _idToToken.ContainsKey(id) ? _idToToken[id] : "<unk>"));

//    }

//    /*
//        def encode(self, text):
//            bpe_tokens = []
//            for token in re.findall(self.pat, text):
//                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
//                bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
//            return bpe_tokens
//    */

//    internal override int[] Encode(string prompt)
//    {
//        // Implement BPE encoding logic here
//        throw new NotImplementedException("BPE encoding not implemented.");
//    }
//}