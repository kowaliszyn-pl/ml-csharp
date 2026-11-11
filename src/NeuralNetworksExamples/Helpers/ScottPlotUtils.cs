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
        string[]? plotWords,
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

            if (plotWords == null || plotWords.Contains(word))
            {
                double x = points[i, 0];
                double y = points[i, 1];

                Scatter scatter = plt.Add.Scatter(x, y);
                scatter.MarkerSize = 5;
                scatter.MarkerShape = MarkerShape.FilledCircle;

                Text text = plt.Add.Text(word, x, y);
                text.LabelAlignment = Alignment.MiddleLeft;
                text.LabelOffsetX = 10;
            }
        }
        SavePlotAsPng(plt, title, xLabel, yLabel, xSize, ySize, outputPath);
    }

    private static void SavePlotAsPng(Plot plt, string title, string xLabel, string yLabel, int xSize, int ySize, string outputPath)
    {
        plt.Title(title);
        plt.XLabel(xLabel);
        plt.YLabel(yLabel);
        plt.SavePng(outputPath, xSize, ySize);
    }

    public static void SaveWordVecGroupPointsAsPng(
        float[,] points,
        List<string> words,
        List<List<string>> plotWordsInGroups,
        List<string> groups,
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
        for (int groupIndex = 0; groupIndex < plotWordsInGroups.Count; groupIndex++)
        {
            List<double> xPoints = [];
            List<double> yPoints = [];

            List<string> wordsInGroup = plotWordsInGroups[groupIndex];
            string groupName = groups[groupIndex];
            for (int i = 0; i < rows; i++)
            {
                string word = words[i];
                if (wordsInGroup.Contains(word))
                {
                    xPoints.Add(points[i, 0]);
                    yPoints.Add(points[i, 1]);

                    Text text = plt.Add.Text(word, points[i, 0], points[i, 1]);
                    text.LabelAlignment = Alignment.UpperCenter;
                    text.LabelOffsetY = -28;
                }
            }

            Scatter scatter = plt.Add.ScatterPoints(xPoints, yPoints);
            scatter.LegendText = groupName;
            scatter.MarkerSize = 20;
        }
        plt.ShowLegend(Alignment.UpperLeft);
        SavePlotAsPng(plt, title, xLabel, yLabel, xSize, ySize, outputPath);
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
        SavePlotAsPng(plt, title, xLabel, yLabel, xSize, ySize, outputPath);
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
