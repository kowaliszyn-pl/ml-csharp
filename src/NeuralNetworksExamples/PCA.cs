// Neural Networks in C♯
// File name: PCA.cs
// www.kowaliszyn.pl, 2025 - 2026

using Accord.Statistics.Analysis;

namespace NeuralNetworksExamples;

class PCA
{
    static void Main()
    {
        // 1. Twoje dane: Tablica embeddingów wysokowymiarowych (np. 5 wektorów po 4 wymiary każdy)
        double[][] embeddings = new double[][]
        {
            new double[] { 0.23, 0.89, -0.45, 0.12 },
            new double[] { 0.25, 0.84, -0.44, 0.10 },
            new double[] { -0.71, 0.11, 0.65, -0.32 },
            new double[] { -0.68, 0.15, 0.60, -0.30 },
            new double[] { 0.05, -0.55, 0.12, 0.88 }
        };

        // 2. Inicjalizacja PCA (ustawiamy docelową liczbę komponentów na 2)
        var pca = new PrincipalComponentAnalysis()
        {
            Method = PrincipalComponentMethod.CovarianceMatrix
        };

        // 3. Analiza danych wejściowych i dopasowanie modelu
        pca.Learn(embeddings);

        // Upewniamy się, że zachowujemy dokładnie 2 pierwsze komponenty dla wizualizacji 2D
        pca.NumberOfOutputs = 2;

        // 4. Transformacja (projekcja) embeddingów do przestrzeni 2D
        double[][] coords2D = pca.Transform(embeddings);

        // 5. Wyświetlenie wyników (współrzędne X, Y gotowe do wykresu)
        for (int i = 0; i < coords2D.Length; i++)
        {
            Console.WriteLine($"Embedding {i} -> X: {coords2D[i][0]:F4}, Y: {coords2D[i][1]:F4}");
        }
    }
}
