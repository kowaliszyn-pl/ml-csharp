// Neural Networks in C♯
// File name: DataSource.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.DataSources;

public abstract class DataSource<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    public abstract (TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) GetData();
}
