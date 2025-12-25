// Neural Networks in C♯
// File name: PredictionDelegate.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Models;

namespace NeuralNetworks.Trainers;

public delegate float EvalFunction<TInputData, TPrediction>(
    Model<TInputData, TPrediction> model,
    TInputData xEvalTest,
    TPrediction yEvalTest,
    TPrediction? predictionLogits
)
    where TInputData : notnull
    where TPrediction : notnull;
