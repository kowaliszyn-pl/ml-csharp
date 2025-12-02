// Neural Networks in C♯
// File name: DataSource.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.DataSources;

public abstract class DataSource<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    /// <summary>
    /// Gets the data for training and testing.
    /// </summary>
    /// <returns>
    /// A tuple containing the training and (optional) testing data.
    /// </returns>
    public abstract (TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) GetData();
}
