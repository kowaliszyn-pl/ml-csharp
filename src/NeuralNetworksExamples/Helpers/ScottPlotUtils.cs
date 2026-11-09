// Neural Networks in C♯
// File name: ScottPlotUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using ScottPlot;
using ScottPlot.Plottables;
using ScottPlot.Statistics;

namespace NeuralNetworksExamples.Helpers;

internal static class ScottPlotUtils
{
    public static void SaveWordvecPointsAsPng(
        float[,] points,
        List<string> words,
        List<string>? printWords,
        string title,
        string xLabel,
        string yLabel,
        int xSize,
        int ySize,
        string outputPath
    )
    {
        using Plot plt = new();
        int rows = points.GetLength(0);
        for (int i = 0; i < rows; i++)
        {
            string word = words[i];

            if (printWords == null || printWords.Contains(word))
            {
                double x = points[i, 0];
                double y = points[i, 1];

                Scatter scatter = plt.Add.Scatter(new double[] { x }, new double[] { y });
                scatter.MarkerSize = 5;
                scatter.MarkerShape = MarkerShape.FilledCircle;

                var text = plt.Add.Text(word, x, y);
                text.LabelAlignment = Alignment.MiddleLeft;
                text.LabelOffsetX = 10;
                // scatter.LegendText = word;
                // Add text label for the word
                // plt.AddText(word, x, y, size: 12, color: System.Drawing.Color.Black);
            }
        }
        plt.Title(title);
        plt.XLabel(xLabel);
        plt.YLabel(yLabel);
        plt.SavePng(outputPath, xSize, ySize);
    }

    public static void SaveDigitPointsAsPng(
        float[,] points,
        float[,] labels,
        string title,
        string xLabel,
        string yLabel,
        int xSize,
        int ySize,
        string outputPath)
    {
        using Plot plt = new();
        int rows = points.GetLength(0);

        // Group points by digit
        for (int digit = 0; digit <= 9; digit++)
        {
            List<double> xPoints = [];
            List<double> yPoints = [];

            for (int i = 0; i < rows; i++)
            {
                if ((int)labels[i, 0] == digit)
                {
                    xPoints.Add(points[i, 0]);
                    yPoints.Add(points[i, 1]);
                }
            }

            Scatter scatter = plt.Add.ScatterPoints(xPoints, yPoints);
            scatter.LegendText = $"Digit {digit}";
            scatter.MarkerSize = 5;
        }

        plt.ShowLegend();
        plt.Title(title);
        plt.XLabel(xLabel);
        plt.YLabel(yLabel);

        plt.SavePng(outputPath, xSize, ySize);
    }

    public static void SaveHistogramAsPng(
        float[,] data,
        int binCount,
        string title,
        string xLabel,
        string yLabel,
        int xSize,
        int ySize,
        string outputPath
    )
    {
        double[] allValues = new double[data.Length];
        int index = 0;
        for (int i = 0; i < data.GetLength(0); i++)
        {
            for (int j = 0; j < data.GetLength(1); j++)
            {
                allValues[index++] = data[i, j];
            }
        }

        Histogram histogram = Histogram.WithBinCount(binCount, allValues);
        using Plot pltHistogram = new();

        // add histogram to the plot
        Bar[] bars = [.. Enumerable.Range(0, histogram.Counts.Length)
            .Select(i => new Bar
            {
                Position = histogram.Bins[i] + histogram.FirstBinSize / 2, // center of bin
                Value = histogram.Counts[i],
                Size = histogram.FirstBinSize
            })];

        pltHistogram.Add.Bars(bars);

        pltHistogram.Title(title);
        pltHistogram.XLabel(xLabel);
        pltHistogram.YLabel(yLabel);

        pltHistogram.SavePng(outputPath, xSize, ySize);
    }
}
