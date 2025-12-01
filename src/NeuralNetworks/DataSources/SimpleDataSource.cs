// Neural Networks in C♯
// File name: SimpleDataSource.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.DataSources;

public class SimpleDataSource<TInputData, TPrediction>(TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) : DataSource<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    public override (TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) GetData() => (xTrain, yTrain, xTest, yTest);
}
