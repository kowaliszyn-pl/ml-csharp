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
    : BaseModel<int[,], float[,]>(new SoftmaxCrossEntropyLoss(), random)
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
    /// <returns>Embedding matrix [batchSize * embeddingDim].</returns>
    public float[,] GetEmbeddingOutput()
    {
        // The embeddings are stored in the first layer (EmbeddingLayer)
        return _embeddingLayer?.Output
            ?? throw new InvalidOperationException("Embedding layer output is not available. Ensure the model has been trained or the embedding layer has been initialized.");
    }

    /// <returns>Embedding matrix [vocabSize * embeddingDim].</returns>
    public float[,] GetEmbeddings()
    {
        // The embeddings are stored in the first layer (EmbeddingLayer)
        return _embeddingLayer?.GetEmbeddings()
            ?? throw new InvalidOperationException("Embedding layer is not initialized.");
    }
}

internal class Word2Vec
{
    private const int RandomSeed = 42;
    private const int Epochs = 220;
    private const int BatchSize = 32;
    private const int EvalEveryEpochs = 40;
    private const int LogEveryEpochs = 20;
    private const int EmbeddingDim = 3;
    private const int WindowSize = 2; // Context window size

    private const float InitialLearningRate = 1e-2f;
    private const float FinalLearningRate = 9e-5f;

    public static void Run()
    {
        ILogger logger = Program.LoggerFactory.CreateLogger<Word2Vec>();

        AnsiConsole.MarkupLine("[bold cyan]═══════════════════════════════════════[/]");
        AnsiConsole.MarkupLine("[bold cyan]   Word2Vec - Skip-gram Model[/]");
        AnsiConsole.MarkupLine("[bold cyan]═══════════════════════════════════════[/]\n");

        // Sample corpus for demonstration
        string[] corpus =
        [
            "the quick brown fox jumps over the lazy dog",
            "the dog sleeps in the sun",
            "the cat climbs the tree",
            "a quick fox runs fast",
            "the brown dog barks loud",
            "lazy cat sleeps all day",
            "the sun shines bright",
            "quick brown animals run fast",
            "the fox and the dog play together",
            "brown cat sits on the tree",
            "dog and fox are animals",
            "the sun sets in the west",
            "cat and dog have four legs",
            "roman eats pizza",
            "bogna likes to read books",
            "roman family goes to the park",
            "the cat and the dog are friends",
            "roman loves his brown dog",
            "bogna has a black cat",
            "roman and bogna have a happy family",
            "every family has a pet",
            "east and west are directions",
            "dog and cat are common pets",
            "cat eats fish and dog eats meat",
            "fish are found in water",
            "pizza is a popular food",
            "pizza and meat are enjoyed by roman",
            "bogna and roman are friends",
        ];

        // Build vocabulary
        Dictionary<string, int> wordToIndex = [];
        List<string> indexToWord = [];

        foreach (string sentence in corpus)
        {
            foreach (string word in sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                if (!wordToIndex.ContainsKey(word))
                {
                    wordToIndex[word] = indexToWord.Count;
                    indexToWord.Add(word);
                }
            }
        }

        int vocabSize = indexToWord.Count;
        AnsiConsole.MarkupLine($"[green]✓ Vocabulary size: {vocabSize}[/]");
        AnsiConsole.MarkupLine($"[dim]Words: {string.Join(", ", indexToWord.Order())}[/]\n");

        // Generate training pairs (center word, context word) using Skip-gram
        List<(int center, int context)> trainingPairs = [];

        foreach (string sentence in corpus)
        {
            string[] words = sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int[] indices = words.Select(w => wordToIndex[w]).ToArray();

            for (int i = 0; i < indices.Length; i++)
            {
                int centerWord = indices[i];

                // Get context words within window
                for (int j = Math.Max(0, i - WindowSize); j <= Math.Min(indices.Length - 1, i + WindowSize); j++)
                {
                    if (i != j)
                    {
                        trainingPairs.Add((centerWord, indices[j]));
                    }
                }
            }
        }

        AnsiConsole.MarkupLine($"[green]✓ Training pairs generated: {trainingPairs.Count}[/]\n");

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

        // Create simple data source (using a small subset for validation)

        SimpleDataSource<int[,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);

        SeededRandom commonRandom = new(RandomSeed);

        // Create Word2Vec model
        Word2VecModel model = new(vocabSize, EmbeddingDim, commonRandom);

        // Create optimizer with learning rate schedule
        ExponentialDecayLearningRate learningRateSchedule = new(InitialLearningRate, FinalLearningRate, Epochs);
        AdamOptimizer optimizer = new(learningRateSchedule);

        // Create trainer
        Trainer<int[,], float[,]> trainer = new(model, optimizer, ConsoleOutputMode.OnlyOnEval, commonRandom, logger);

        AnsiConsole.MarkupLine("[bold yellow]🚀 Training Word2Vec model...[/]\n");

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

        AnsiConsole.MarkupLine("\n[bold green]✓ Training completed![/]\n");

        // Demonstrate word similarity
        DemonstrateWordSimilarity(model, indexToWord, wordToIndex);

        DisplayAllWordsWithEmbeddings(model, indexToWord, wordToIndex);
    }

    private static void DisplayAllWordsWithEmbeddings(Word2VecModel model, List<string> indexToWord, Dictionary<string, int> wordToIndex)
    {
        int[,] xWords = new int[indexToWord.Count, 1];
        for (int i = 0; i < indexToWord.Count; i++)
        {
            xWords[i, 0] = i;
        }

        _ = model.Forward(xWords, inference: true);

        float[,] embeddings = model.GetEmbeddingOutput();

        Table table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Cyan1)
            .AddColumn(new TableColumn("[bold]Word[/]").Centered())
            .AddColumn(new TableColumn("[bold]Embedding Vector[/]").LeftAligned());

        for (int i = 0; i < indexToWord.Count; i++)
        {
            string word = indexToWord[i];
            string embeddingVector = string.Join(", ", Enumerable.Range(0, EmbeddingDim).Select(j => embeddings[i, j].ToString("F4")));
            table.AddRow($"[yellow]{word}[/]", embeddingVector);
        }

        AnsiConsole.Write(table);
    }

    private static void DemonstrateWordSimilarity(Word2VecModel model, List<string> indexToWord, Dictionary<string, int> wordToIndex)
    {
        AnsiConsole.MarkupLine("[bold cyan]═══════════════════════════════════════[/]");
        AnsiConsole.MarkupLine("[bold cyan]   Word Similarity Demo[/]");
        AnsiConsole.MarkupLine("[bold cyan]═══════════════════════════════════════[/]\n");

        // Test some word similarities
        string[] testWords = ["dog", "cat", "fox", "quick", "lazy", "brown", "roman", "bogna", "family", "pet"];

        Table table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Cyan1)
            .AddColumn(new TableColumn("[bold]Query Word[/]").Centered())
            .AddColumn(new TableColumn("[bold]Most Similar Words[/]").LeftAligned());

        foreach (string word in testWords)
        {
            if (wordToIndex.TryGetValue(word, out int wordIdx))
            {
                // Get embedding for the test word
                int[,] input = new int[1, 1];
                input[0, 0] = wordIdx;

                _ = model.Forward(input, inference: true);
                float[,] embeddings = model.GetEmbeddingOutput();

                // Find most similar words
                List<string> similarWords = FindSimilarWords(embeddings, model, indexToWord, wordToIndex, wordIdx, topK: 3);

                table.AddRow($"[yellow]{word}[/]", string.Join(", ", similarWords));
            }
        }

        AnsiConsole.Write(table);
    }

    private static List<string> FindSimilarWords(float[,] queryEmbeddings, Word2VecModel model,
        List<string> indexToWord, Dictionary<string, int> wordToIndex, int queryIdx, int topK)
    {
        // Compute cosine similarity with all words
        List<(string word, float similarity)> similarities = [];

        /* TODO: This performs a full forward pass per vocabulary entry (`O(vocabSize)` forwards) for each query word, which will become a bottleneck once vocab grows. Prefer pulling the embedding matrix once (or caching all embeddings) and computing cosine similarity directly against vectors, avoiding repeated model forwards.
         * */

        for (int i = 0; i < indexToWord.Count; i++)
        {
            if (i == queryIdx) continue; // Skip the query word itself

            int[,] input = new int[1, 1];
            input[0, 0] = i;
            _ = model.Forward(input, inference: true);
            float[,] embeddings = model.GetEmbeddingOutput();

            float similarity = CosineSimilarity(queryEmbeddings, embeddings);
            similarities.Add((indexToWord[i], similarity));
        }

        return similarities
            .OrderByDescending(x => x.similarity)
            .Take(topK)
            .Select(x => $"[green]{x.word}[/] [dim]({x.similarity:F3})[/]")
            .ToList();
    }

    private static float CosineSimilarity(float[,] a, float[,] b)
    {
        int dim = a.GetLength(1);
        float dotProduct = 0;
        float normA = 0;
        float normB = 0;

        for (int i = 0; i < dim; i++)
        {
            float valA = a[0, i];
            float valB = b[0, i];
            dotProduct += valA * valB;
            normA += valA * valA;
            normB += valB * valB;
        }

        float denominator = MathF.Sqrt(normA) * MathF.Sqrt(normB);
        return denominator > 0 ? dotProduct / denominator : 0;
    }
}
