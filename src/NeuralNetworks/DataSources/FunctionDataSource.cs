// Machine Learning Utils
// File name: FunctionDataSource.cs
// Code It Yourself with .NET, 2024

using NeuralNetworks.Core;

namespace NeuralNetworks.DataSources;

public class FunctionDataSource(float[,] arguments, Func<float[], float> function, float testRation = 0.7f, SeededRandom? random = null) 
    : DataSource<float[,], float[,]>
{
    public override (float[,] xTrain, float[,] yTrain, float[,]? xTest, float[,]? yTest) GetData()
    {
        arguments.PermuteInPlace(random);
        int argRows = arguments.GetLength(0);
        int argColumns = arguments.GetLength(1);

        float[,] yData = new float[argRows, 1];

        for (int row = 0; row < argRows; row++)
        {
            float[] rowData = new float[argColumns];
            for (int column = 0; column < argColumns; column++)
            {
                rowData[column] = arguments[row, column];
            }
            yData[row, 0] = function(rowData);
        }

        (float[,] xTrain, float[,]? xTest) = arguments.SplitRowsByRatio(testRation);
        (float[,] yTrain, float[,]? yTest) = yData.SplitRowsByRatio(testRation);

        return (xTrain, yTrain, xTest, yTest);  
    }
}
