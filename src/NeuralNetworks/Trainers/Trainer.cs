// Neural Networks in C♯
// File name: Trainer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.DataSources;
using NeuralNetworks.Layers;
using NeuralNetworks.Models;
using NeuralNetworks.Optimizers;
using NeuralNetworks.Trainers.Logging;

using static System.Console;

namespace NeuralNetworks.Trainers;

/// <summary>
/// Represents a trainer for a neural network.
/// </summary>
public abstract class Trainer<TInputData, TPrediction>(
    Model<TInputData, TPrediction> model,
    Optimizer optimizer,
    ConsoleOutputMode consoleOutputMode = ConsoleOutputMode.OnlyOnEval,
    SeededRandom? random = null,
    ILogger<Trainer<TInputData, TPrediction>>? logger = null)
    where TInputData : notnull
    where TPrediction : notnull
{
    private float _bestLoss = float.MaxValue;

    /// <summary>
    /// Gets or sets the memo associated with the trainer.
    /// </summary>
    public string? Memo { get; set; }

    /// <summary>
    /// Generates batches of input and output matrices.
    /// </summary>
    /// <param name="x">The input matrix.</param>
    /// <param name="y">The output matrix.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>An enumerable of batches.</returns>
    protected abstract IEnumerable<(TInputData xBatch, TPrediction yBatch)> GenerateBatches(TInputData x, TPrediction y, int batchSize = 32);

    protected abstract (TInputData, TPrediction) PermuteData(TInputData x, TPrediction y, Random random);

    protected abstract float GetRows(TInputData x);

    /// <summary>
    /// Fits the neural network to the provided data source.
    /// </summary>
    /// <param name="dataSource">The data source.</param>
    /// <param name="evalFunction">The evaluation function.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="evalEveryEpochs">The number of epochs between evaluations.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="restart">A flag indicating whether to restart the training.</param>
    [SuppressMessage("Usage", "CA2254:Template should be a static expression", Justification = "<Pending>")]
    [SuppressMessage("Performance", "CA1873:Avoid potentially expensive logging", Justification = "<Pending>")]
    public void Fit(
        DataSource<TInputData, TPrediction> dataSource,
        Func<Model<TInputData, TPrediction>, TInputData, TPrediction, float>? evalFunction = null,
        int epochs = 100,
        int evalEveryEpochs = 10,
        int logEveryEpochs = 1,
        int batchSize = 32,
        bool earlyStop = false,
        bool restart = true)
    {
        Stopwatch trainWatch = Stopwatch.StartNew();

        logger?.LogInformation(string.Empty);
        logger?.LogInformation("===== Begin Log =====");
        logger?.LogInformation("Fit started with params: epochs: {epochs}, evalEveryEpochs: {evalEveryEpochs}, logEveryEpochs: {logEveryEpochs}, batchSize: {batchSize}, optimizer: {optimizer}, random: {random}.", epochs, evalEveryEpochs, logEveryEpochs, batchSize, optimizer, random);
        logger?.LogInformation("Model type: {modelType}.", model.GetType().Name);
        logger?.LogInformation("Model layers:");
        foreach (Layer layer in model.Layers)
        {
            logger?.LogInformation("Layer: {layer}.", layer);
        }
        logger?.LogInformation("Loss function: {loss}", model.LossFunction);

        if (Memo is not null)
            logger?.LogInformation("Memo: \"{memo}\".", Memo);

#if DEBUG
        string environment = "Debug";
#else
        string environment = "Release";
#endif
        logger?.LogInformation("Environment: {environment}.", environment);

        (TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) = dataSource.GetData();
        int allSteps = (int)Math.Ceiling(GetRows(xTrain) / (float)batchSize);

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            bool lastEpoch = epoch == epochs;
            bool evaluationEpoch = epoch % evalEveryEpochs == 0 || lastEpoch;
            bool logEpoch = epoch % logEveryEpochs == 0 || lastEpoch;

            if(logEpoch)
                logger?.LogInformation("Epoch {epoch}/{epochs} started.", epoch, epochs);

            bool eval = xTest is not null && yTest is not null && evaluationEpoch;

            if ((logEpoch && consoleOutputMode == ConsoleOutputMode.OnlyOnEval) || consoleOutputMode == ConsoleOutputMode.Always)
                WriteLine($"\nEpoch {epoch}/{epochs}...");

            // Epoch should be later than 1 to save the first checkpoint.
            //if (eval && epoch > 1)
            //{
            //    model.SaveCheckpoint();
            //    logger?.LogInformation("Checkpoint saved.");
            //}

            (xTrain, yTrain) = PermuteData(xTrain, yTrain, random ?? new Random());
            optimizer.UpdateLearningRate(epoch, epochs);

            if(logEpoch)
                WriteLine($"CurrentLearningRate: {optimizer.LearningRate.GetLearningRate()}.");

            float? trainLoss = null;
            int step = 0;

            float? stepsPerSecond = null;

            Stopwatch stepWatch = Stopwatch.StartNew();
            foreach ((TInputData xBatch, TPrediction yBatch) in GenerateBatches(xTrain, yTrain, batchSize))
            {
                step++;
                if (allSteps > 1 && consoleOutputMode > ConsoleOutputMode.Disable)
                {
                    string stepInfo = $"Step {step}/{allSteps}/{epoch}/{epochs}...";
                    if (stepsPerSecond is not null)
                        stepInfo += $" {stepsPerSecond.Value:F2} steps/s";
                    Write(stepInfo + "\r");
                }

                trainLoss = (trainLoss ?? 0) + model.TrainBatch(xBatch, yBatch);
                //optimizer.Step(model);
                model.UpdateParams(optimizer);

                long elapsedMsPerStep = stepWatch.ElapsedMilliseconds / step;
                stepsPerSecond = 1000.0f / elapsedMsPerStep;
            }
            stepWatch.Stop();

            // Write a line with 80 spaces to clean the line with the step info.
            if (allSteps > 1 && consoleOutputMode > ConsoleOutputMode.Disable)
                Write(new string(' ', 80) + "\r");

            if (trainLoss is not null && logEpoch)
            {
                if (consoleOutputMode > ConsoleOutputMode.Disable)
                    WriteLine($"Train loss (average): {trainLoss.Value / allSteps}");
                logger?.LogInformation("Train loss (average): {trainLoss} for epoch {epoch}.", trainLoss.Value / allSteps, epoch);
            }

            if (eval)
            {
                TPrediction testPredictions = model.Forward(xTest!, true);
                float loss = model.LossFunction.Forward(testPredictions, yTest!);

                if (consoleOutputMode > ConsoleOutputMode.Disable)
                    WriteLine($"Test loss: {loss}");
                logger?.LogInformation("Test loss: {testLoss} for epoch {epoch}.", loss, epoch);

                if (evalFunction is not null)
                {
                    float evalValue = evalFunction(model, xTest!, yTest!);

                    if (consoleOutputMode > ConsoleOutputMode.Disable)
                        WriteLine($"Eval: {evalValue:P2}");
                    logger?.LogInformation("Eval: {evalValue:P2} for epoch {epoch}.", evalValue, epoch);
                }

                if (loss < _bestLoss)
                {
                    _bestLoss = loss;
                }
                else if (earlyStop)
                {
                    if (model.HasCheckpoint())
                    {
                        model.RestoreCheckpoint();
                        logger?.LogInformation("Checkpoint restored.");
                    }

                    if (consoleOutputMode > ConsoleOutputMode.Disable)
                        WriteLine($"Early stopping, loss {loss} is greater than {_bestLoss}");
                    logger?.LogInformation("Early stopping. Loss {loss} is greater than {bestLoss}.", loss, _bestLoss);

                    break;
                }

            }
        }

        trainWatch.Stop();

        double elapsedSeconds = trainWatch.Elapsed.TotalSeconds;
        logger?.LogInformation("Fit finished in {elapsedSecond:F2} s.", elapsedSeconds);
        int paramCount = model.GetParamCount();
        logger?.LogInformation("{paramCount:n0} parameters trained.", paramCount);

        if (consoleOutputMode > ConsoleOutputMode.Disable)
        {
            ForegroundColor = ConsoleColor.Cyan;
            WriteLine($"\nFit finished in {elapsedSeconds:F2} s.");
            WriteLine($"{paramCount:n0} parameters trained.");
            ForegroundColor = ConsoleColor.Yellow;
            TPrediction testPredictions = model.Forward(xTest!, true);
            float loss = model.LossFunction.Forward(testPredictions, yTest!);
            WriteLine($"\nLoss on test data: {loss:F5}");
            if(evalFunction is not null)
            {
                float evalValue = evalFunction(model, xTest!, yTest!);
                WriteLine($"Eval on test data: {evalValue:P2}");
            }
            ResetColor();
            WriteLine();
        }

        logger?.LogInformation("===== End Log =====");
        logger?.LogInformation(string.Empty);
    }


}
