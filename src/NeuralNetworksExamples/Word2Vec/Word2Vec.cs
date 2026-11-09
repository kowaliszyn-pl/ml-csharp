// Neural Networks in C♯
// File name: Word2Vec.cs
// www.kowaliszyn.pl, 2025 - 2026

using Accord.MachineLearning;

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

using static NeuralNetworksExamples.Helpers.AccordUtils;
using static NeuralNetworksExamples.Helpers.ScottPlotUtils;

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

        // Remove 'the', 'a', 'and', 'in', 'on', 'to', 'is', 'are', 'has', 'have', 'of', 'for', 'with' from the corpus
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
        Console.WriteLine($"First 15 training pairs: {string.Join(", ", trainingPairs.Take(15).Select(p => $"({tokenIdToWord[p.center]}, {tokenIdToWord[p.context]})"))}\n");

        // Create simple data source (using a small subset for validation)

        SimpleDataSource<int[,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);

        SeededRandom commonRandom = new(RandomSeed);

        // Create Word2Vec model
        Word2VecModel model = new(vocabSize, EmbeddingDim, commonRandom);

        // Create optimizer with learning rate schedule
        ExponentialDecayLearningRate learningRateSchedule = new(InitialLearningRate, FinalLearningRate);
        AdamOptimizer optimizer = new(learningRateSchedule);

        // Create trainer
        Trainer<int[,], float[,]> trainer = new(
            model,
            optimizer,
            ConsoleOutputMode.OnlyOnEval,
            commonRandom,
            logger
        );

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

        // Apply dimensionality reduction (PCA) and save the visualization
        SaveEmbeddingsWithPCA(learnedEmbeddings, tokenIdToWord);
    }

    private static void SaveEmbeddingsWithPCA(float[,] embeddings, List<string> tokenIdToWord)
    {
        /*
        List<string> wordsToTransform = ["king", "queen", "man", "woman", "boy", "girl", "male", "female", "3", "iii", "9", "ix", "daughter", "son", "2", "4", "ii", "iv", "three", "nine", "two", "four"];

        float[,] embeddingsToTransform = new float[wordsToTransform.Count, EmbeddingDim];
        int rows = embeddings.GetLength(0);
        List<string> tokenIdToWordToTransform = new();

        for (int i = 0; i < rows; i++)
        {
            if(wordsToTransform.Contains(tokenIdToWord[i]))
            {
                int index = wordsToTransform.IndexOf(tokenIdToWord[i]);
                for (int j = 0; j < EmbeddingDim; j++)
                {
                    embeddingsToTransform[index, j] = embeddings[i, j];
                }
                tokenIdToWordToTransform.Add(tokenIdToWord[i]);
            }
        }*/

        // Reduce dimensions to 2D using PCA
        float[,] reducedEmbeddings = TransformWithPCA(embeddings, outputDimensions: 2); // outputDimensions: 2

        /*
        // Display the reduced embeddings
        Console.WriteLine("\n2D Embeddings (after PCA):");
        for (int i = 0; i < tokenIdToWord.Count; i++)
        {
            Console.WriteLine($"Word: {tokenIdToWord[i]}, X: {reducedEmbeddings[i, 0]:F4}, Y: {reducedEmbeddings[i, 1]:F4}");
        }*/

        SaveWordvecPointsAsPng(
            reducedEmbeddings,
            tokenIdToWord,
            // ["king", "queen", "man", "woman", "boy", "girl", "male", "female", "3", "iii", "9", "ix", "daughter", "son", "2", "4", "ii", "iv", "three", "nine", "two", "four", "sweden", "stockholm", "copenhagen", "denmark", "england", "london", "warsaw", "poland"],
            ["king", "queen", "man", "woman", "boy", "girl", "male", "female", "daughter", "son", "sweden", "stockholm", "copenhagen", "denmark", "england", "london", "warsaw", "poland", "3", "iii", "9", "ix"],
            title: "Word2Vec Embeddings (PCA)",
            xLabel: "PCA 1",
            yLabel: "PCA 2",
            xSize: 1200,
            ySize: 900,
            outputPath: "word2vec_embeddings_pca.png"
        );
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

        /*
         *  Produces:
         *  
         *  Word Similarities
         *  king ≈ queen (0,882), continent (0,670), royal (0,480), young (0,414)
         *  queen ≈ king (0,882), continent (0,583), rules (0,487), both (0,453)
         *  man ≈ woman (0,975), boy (0,798), young (0,752), son (0,687)
         *  woman ≈ man (0,975), young (0,756), boy (0,711), son (0,617)
         *  boy ≈ son (0,835), man (0,798), woman (0,711), daughter (0,706)
         *  girl ≈ daughter (0,912), margrethe (0,791), elizabeth (0,707), two (0,665)
         */

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
            ("iii", "3", "9"),
            ("country", "poland", "america")
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

        /* Produces:
         * 
         * Word Analogies
         * king - man + woman ≈ queen (0,820), continent (0,778), not (0,709), country (0,586)
         * queen - woman + man ≈ king (0,780), royal (0,609), young (0,549), family (0,448)
         * man - boy + girl ≈ female (0,810), margrethe (0,733), elizabeth (0,651), ii (0,592)
         * woman - girl + boy ≈ man (0,760), state (0,505), royal (0,487), three (0,487)
         * sweden - stockholm + copenhagen ≈ denmark (0,972), rulers (0,729), called (0,686), england (0,539)
         * england - london + warsaw ≈ poland (0,888), ruler (0,830), who (0,827), denmark (0,756)
         * iii - 3 + 9 ≈ ix (0,989), frederik (0,815), nine (0,587), australia (0,567)
         */
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

        float denominator = MathF.Sqrt(normA) * MathF.Sqrt(normB);
        return denominator > 0 ? dotProduct / denominator : 0;
    }
}
