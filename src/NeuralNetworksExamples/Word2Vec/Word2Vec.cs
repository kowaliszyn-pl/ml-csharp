// Neural Networks in C♯
// File name: Word2Vec.cs
// www.kowaliszyn.pl, 2025 - 2026

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.DataSources;
using NeuralNetworks.Layers;
using NeuralNetworks.LearningRates;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.Optimizers;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Trainers;
using NeuralNetworks.Trainers.Logging;

using Spectre.Console;

namespace NeuralNetworksExamples.Word2Vec;

/// <summary>
/// Word2Vec model using Skip-gram architecture.
/// Maps center words to context words.
/// </summary>
internal class Word2VecModel(int vocabSize, int embeddingDim, SeededRandom? random = null)
    : BaseModel<int[,], float[,]>(new LogSoftmaxCrossEntropyLoss(), random)
{
    private EmbeddingLayer? _embeddingLayer;

    protected override LayerListBuilder<int[,], float[,]> CreateLayerListBuilder()
    {
        GlorotInitializer initializer = new(Random);
        return
            AddLayer(_embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim, initializer))
            .AddLayer(new DenseLayer(vocabSize, new Linear(), initializer));
    }

    /// <summary>
    /// Gets the embeddings for the last batch.
    /// </summary>
    /// <returns>Embedding matrix [batchSize, embeddingDim].</returns>
    public float[,] GetEmbeddingOutput()
    {
        // The embeddings are stored in the first layer (EmbeddingLayer)
        return _embeddingLayer?.Output
            ?? throw new InvalidOperationException("Embedding layer output is not available. Ensure the model has been trained or the embedding layer has been initialized.");
    }

    /// <returns>Embedding matrix [vocabSize, embeddingDim].</returns>
    public float[,] GetEmbeddings()
    {
        // The embeddings are stored in the first layer (EmbeddingLayer)
        return _embeddingLayer?.GetEmbeddings()
            ?? throw new InvalidOperationException("Embedding layer is not initialized.");
    }
}

internal class Word2Vec
{
    private const int RandomSeed = 260720;
    private const int Epochs = 1000; // 2000;
    private const int BatchSize = 16;
    private const int EvalEveryEpochs = 200;
    private const int LogEveryEpochs = 100;
    private const int EmbeddingDim = 7;
    private const int WindowSize = 3; // Context window size
    private const string DataFileName = "data1.txt";

    private const float InitialLearningRate = 0.1f;
    private const float FinalLearningRate = 0.01f;

    public static void Run()
    {
        ILogger logger = Program.LoggerFactory.CreateLogger<Word2Vec>();

        Console.WriteLine("Load and preprocess the corpus...");

        // Sample corpus for demonstration
        string[] corpus = [.. File.ReadAllLines(Path.Combine(Program.Word2VecDataFolderPath, DataFileName))
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(line => line.Trim().ToLower())];

        // Build vocabulary
        Dictionary<string, int> wordToTokenId = [];
        List<string> tokenIdToWord = [];

        // Remove 'the', 'a', 'and', 'in', 'on', 'to', 'is', 'are', 'has', 'have', 'of', 'for', 'with' from corpus
        HashSet<string> stopWords = ["the", "a", "an", "and", "in", "on", "to", "is", "are", "has", "have", "of", "for", "with", "s", "be", "was"];
        char[] separators = [' ', '.', ',', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '-', '_', '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', '\''];

        string[][] words = new string[corpus.Length][];

        for (int i = 0; i < corpus.Length; i++)
        {
            words[i] = [.. corpus[i]
                .Split(separators, StringSplitOptions.RemoveEmptyEntries)
                .Where(word => !stopWords.Contains(word))];
        }

        foreach (string[] sentence in words)
        {
            foreach (string word in sentence)
            {
                if (!wordToTokenId.ContainsKey(word))
                {
                    wordToTokenId[word] = tokenIdToWord.Count;
                    tokenIdToWord.Add(word);
                }
            }
        }

        int vocabSize = tokenIdToWord.Count;

        // Generate training pairs (center word, context word) using Skip-gram
        List<(int center, int context)> trainingPairs = [];

        foreach (string[] sentence in words)
        {
            int[] tokenIdsInSentence = sentence.Select(w => wordToTokenId[w]).ToArray();
            int sentenceLength = tokenIdsInSentence.Length;

            for (int i = 0; i < sentenceLength; i++)
            {
                int centerWord = tokenIdsInSentence[i];

                // Get context words within window
                for (int j = Math.Max(0, i - WindowSize); j <= Math.Min(sentenceLength - 1, i + WindowSize); j++)
                {
                    if (i != j)
                    {
                        trainingPairs.Add((centerWord, tokenIdsInSentence[j]));
                    }
                }
            }
        }

        // Prepare training data
        int numSamples = trainingPairs.Count;
        int[,] xTrain = new int[numSamples, 1]; // Center words
        float[,] yTrain = new float[numSamples, vocabSize]; // One-hot encoded context words

        for (int i = 0; i < numSamples; i++)
        {
            xTrain[i, 0] = trainingPairs[i].center;
            yTrain[i, trainingPairs[i].context] = 1.0f;
        }

        // Create xTest and yTest
        int validationSize = Math.Min(20, numSamples);
        int[,] xTest = new int[validationSize, 1];
        float[,] yTest = new float[validationSize, vocabSize];

        for (int i = 0; i < validationSize; i++)
        {
            xTest[i, 0] = xTrain[i, 0];
            for (int j = 0; j < vocabSize; j++)
            {
                yTest[i, j] = yTrain[i, j];
            }
        }

        Console.WriteLine($"\nVocabulary size: {vocabSize}");
        Console.WriteLine($"Words: {string.Join(", ", tokenIdToWord.Order())}");
        Console.WriteLine($"\nTraining pairs generated: {trainingPairs.Count}");
        Console.WriteLine($"First 5 training pairs: {string.Join(", ", trainingPairs.Take(5).Select(p => $"({tokenIdToWord[p.center]}, {tokenIdToWord[p.context]})"))}\n");

        // Create simple data source (using a small subset for validation)

        SimpleDataSource<int[,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);

        SeededRandom commonRandom = new(RandomSeed);

        // Create Word2Vec model
        Word2VecModel model = new(vocabSize, EmbeddingDim, commonRandom);

        // Create optimizer with learning rate schedule
        ExponentialDecayLearningRate learningRateSchedule = new(InitialLearningRate, FinalLearningRate);
        AdamOptimizer optimizer = new(learningRateSchedule);

        // Create trainer
        Trainer<int[,], float[,]> trainer = new(model, optimizer, ConsoleOutputMode.OnlyOnEval, commonRandom, logger);

        Console.WriteLine("🚀 Training Word2Vec model...\n");

        // Train the model
        trainer.Fit(
            dataSource,
            epochs: Epochs,
            evalEveryEpochs: EvalEveryEpochs,
            logEveryEpochs: LogEveryEpochs,
            batchSize: BatchSize,
            restart: true,
            displayDescriptionOnStart: true
        );

        float[,] learnedEmbeddings = model.GetEmbeddings();

        // Display all words with their embeddings
        //DisplayAllWordsWithEmbeddings(learnedEmbeddings, tokenIdToWord);

        // Demonstrate word similarity
        DemonstrateWordSimilarity(learnedEmbeddings, tokenIdToWord, wordToTokenId);
    }

    private static void DisplayAllWordsWithEmbeddings(float[,] embeddings, List<string> tokenIdToWord)
    {
        int vocabSize = tokenIdToWord.Count;
        int[,] xWords = new int[vocabSize, 1];
        for (int i = 0; i < vocabSize; i++)
        {
            xWords[i, 0] = i;
        }

        Table table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Cyan1)
            .AddColumn(new TableColumn("[bold]Word[/]").Centered())
            .AddColumn(new TableColumn("[bold]Embedding Vector[/]").LeftAligned());

        for (int tokenId = 0; tokenId < vocabSize; tokenId++)
        {
            string word = tokenIdToWord[tokenId];
            string embeddingVector = string.Join(", ", Enumerable.Range(0, EmbeddingDim).Select(j => embeddings[tokenId, j].ToString("F4")));
            table.AddRow($"[yellow]{word}[/]", embeddingVector);
        }

        AnsiConsole.Write(table);
    }

    private static void DemonstrateWordSimilarity(float[,] embeddings, List<string> tokenIdToWord, Dictionary<string, int> wordToTokenId)
    {
        Console.WriteLine("\nWord Similarities");

        // Test some word similarities
        string[] testWords = ["king", "queen", "man", "woman", "boy", "girl", "flowers", "melody", "broken"];

        foreach (string word in testWords)
        {
            if (wordToTokenId.TryGetValue(word, out int tokenId))
            {
                // Find most similar words
                List<string> similarWords = FindSimilarWords(embeddings, tokenIdToWord, tokenId, topK: 4);

                // table.AddRow($"[yellow]{word}[/]", string.Join(", ", similarWords));

                AnsiConsole.MarkupLine($"[yellow]{word}[/] ≈ {string.Join(", ", similarWords)}");
            }
        }

        // Test the following analogies:
        // king - man + woman = ?
        // queen - woman + man = ?
        // man - boy + girl = ?
        // woman - girl + boy = ?
        // sweden - stockholm + copenhagen  = ?
        // iii - 3 + 9 = ?

        Console.WriteLine("\nWord Analogies");

        List<(string wordA, string wordB, string wordC)> analogies = [
            ("king", "man", "woman"),
            ("queen", "woman", "man"),
            ("man", "boy", "girl"),
            ("woman", "girl", "boy"),
            ("sweden", "stockholm", "copenhagen"),
            ("england", "london", "warsaw"),
            ("iii", "3", "9")
        ];

        foreach ((string? wordA, string? wordB, string? wordC) in analogies)
        {
            if (wordToTokenId.TryGetValue(wordA, out int tokenIdA) &&
                wordToTokenId.TryGetValue(wordB, out int tokenIdB) &&
                wordToTokenId.TryGetValue(wordC, out int tokenIdC))
            {
                float[] embeddingA = new float[EmbeddingDim];
                float[] embeddingB = new float[EmbeddingDim];
                float[] embeddingC = new float[EmbeddingDim];
                for (int i = 0; i < EmbeddingDim; i++)
                {
                    embeddingA[i] = embeddings[tokenIdA, i];
                    embeddingB[i] = embeddings[tokenIdB, i];
                    embeddingC[i] = embeddings[tokenIdC, i];
                }
                // Compute the analogy vector: A - B + C
                float[] analogyVector = new float[EmbeddingDim];
                for (int i = 0; i < EmbeddingDim; i++)
                {
                    analogyVector[i] = embeddingA[i] - embeddingB[i] + embeddingC[i];
                }

                // Find the most similar word to the analogy vector
                List<(string word, float similarity)> similarities = [];
                for (int tokenId = 0; tokenId < tokenIdToWord.Count; tokenId++)
                {
                    if (tokenId == tokenIdA || tokenId == tokenIdB || tokenId == tokenIdC)
                        continue; // Skip the words used in the analogy
                    float[] candidateEmbedding = new float[EmbeddingDim];
                    for (int i = 0; i < EmbeddingDim; i++)
                    {
                        candidateEmbedding[i] = embeddings[tokenId, i];
                    }
                    float similarity = CosineSimilarity(analogyVector, candidateEmbedding);
                    similarities.Add((tokenIdToWord[tokenId], similarity));
                }
                (string word, float similarity)[] bestMatches = similarities.OrderByDescending(x => x.similarity).ToArray();
                AnsiConsole.MarkupLine($"[yellow]{wordA}[/] - [yellow]{wordB}[/] + [yellow]{wordC}[/] ≈ [green]{bestMatches[0].word}[/] [dim]({bestMatches[0].similarity:F3})[/], [green]{bestMatches[1].word}[/] [dim]({bestMatches[1].similarity:F3})[/], [green]{bestMatches[2].word}[/] [dim]({bestMatches[2].similarity:F3})[/], [green]{bestMatches[3].word}[/] [dim]({bestMatches[3].similarity:F3})[/]");
            }
        }
    }

    private static List<string> FindSimilarWords(float[,] embeddings, List<string> tokenIdToWord, int queryTokenId, int topK)
    {
        // Compute cosine similarity with all words
        List<(string word, float similarity)> similarities = [];

        float[] queryEmbeddings = new float[EmbeddingDim];
        for (int i = 0; i < EmbeddingDim; i++)
        {
            queryEmbeddings[i] = embeddings[queryTokenId, i];
        }

        for (int tokenId = 0; tokenId < tokenIdToWord.Count; tokenId++)
        {
            if (tokenId == queryTokenId)
                continue; // Skip the query word itself

            float[] candidateEmbedding = new float[EmbeddingDim];
            for (int i = 0; i < EmbeddingDim; i++)
            {
                candidateEmbedding[i] = embeddings[tokenId, i];
            }

            float similarity = CosineSimilarity(queryEmbeddings, candidateEmbedding);
            similarities.Add((tokenIdToWord[tokenId], similarity));
        }

        return [.. similarities
            .OrderByDescending(x => x.similarity)
            .Take(topK)
            .Select(x => $"[green]{x.word}[/] [dim]({x.similarity:F3})[/]")];
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        int dim = a.Length;
        float dotProduct = 0;
        float normA = 0;
        float normB = 0;

        for (int i = 0; i < dim; i++)
        {
            float valA = a[i];
            float valB = b[i];
            dotProduct += valA * valB;
            normA += valA * valA;
            normB += valB * valB;
        }

        if (normA <= 0 || normB <= 0)
            return 0;

        float denominator = MathF.Sqrt(normA) * MathF.Sqrt(normB);
        return denominator > 0 ? dotProduct / denominator : 0;
    }
}
