// Neural Networks in C♯
// File name: Trainer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

using Microsoft.Extensions.Logging;

using NeuralNetworks.Core;
using NeuralNetworks.Core.Operations;
using NeuralNetworks.DataSources;
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
    ILogger<Trainer<TInputData, TPrediction>>? logger = null
)
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

    protected abstract void PermuteData(TInputData x, TPrediction y, Random random);

    protected abstract float GetRows(TInputData x);

    /// <summary>
    /// Fits the neural network to the provided data source.
    /// </summary>
    /// <param name="dataSource">The data source.</param>
    /// <param name="evalFunction">The evaluation function.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="evalEveryEpochs">The number of epochs between evaluations.</param>
    /// <param name="logEveryEpochs">The number of epochs between logging.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="earlyStop">A flag indicating whether to enable early stopping.</param>
    /// <param name="restart">A flag indicating whether to restart the training or continue from the last state.</param>
    /// <param name="displayDescriptionOnStart">A flag indicating whether to display the fit+model description on start.</param>
    /// <param name="operationBackendTimingEnabled">A flag indicating whether to enable operation backend timing. If true, the backend operations timing report will be displayed after training.</param>
    [SuppressMessage("Usage", "CA2254:Template should be a static expression", Justification = "<Pending>")]
    [SuppressMessage("Performance", "CA1873:Avoid potentially expensive logging", Justification = "<Pending>")]
    public void Fit(
        DataSource<TInputData, TPrediction> dataSource,
        EvalFunction<TInputData, TPrediction>? evalFunction = null,
        int epochs = 100,
        int evalEveryEpochs = 10,
        int logEveryEpochs = 1,
        int batchSize = 32,
        bool earlyStop = false,
        bool restart = true,
        bool displayDescriptionOnStart = true,
        bool operationBackendTimingEnabled = false
    )
    {
        try
        {
            Stopwatch trainWatch = Stopwatch.StartNew();

            logger?.LogInformation(string.Empty);
            logger?.LogInformation("===== Begin Log =====");
            logger?.LogInformation("Fit started");

            displayDescriptionOnStart &= consoleOutputMode != ConsoleOutputMode.Disable;

            OperationBackend.StatisticsEnabled = operationBackendTimingEnabled;

            // Describe the trainer configuration
            List<string> description = DescribeFit();
            WriteLineIfLogging();
            foreach (string line in description)
            {
                WriteLineIfLogging(line);
            }
            WriteLineIfLogging();

            void WriteLineIfLogging(string message = "")
            {
                if (displayDescriptionOnStart)
                    WriteLine(message);
                logger?.LogInformation(message);
            }

#if DEBUG
        string environment = "Debug";
#else
            string environment = "Release";
#endif
            logger?.LogInformation("Environment: {environment}.", environment);

            (TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) = dataSource.GetData();
            int allSteps = (int)Math.Ceiling(GetRows(xTrain) / (float)batchSize);
            long allStepsInTraining = allSteps * epochs;

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                bool lastEpoch = epoch == epochs;
                bool evaluationEpoch = epoch % evalEveryEpochs == 0 || lastEpoch;
                bool logEpoch = epoch % logEveryEpochs == 0 || lastEpoch;

                if (logEpoch)
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

                PermuteData(xTrain, yTrain, random ?? new Random());
                optimizer.UpdateLearningRate(1, epoch, epochs);

                if (logEpoch)
                    WriteLine($"Current learning rate: {optimizer.LearningRate.GetLearningRate()}.");

                float? trainLoss = null;
                int step = 0;

                float? stepsPerSecond = null;
                TimeSpan eta;
                long calculatedOn = -500;

                Stopwatch stepWatch = Stopwatch.StartNew();
                foreach ((TInputData xBatch, TPrediction yBatch) in GenerateBatches(xTrain, yTrain, batchSize))
                {
                    step++;

                    // Usually, learning rate is updated once per epoch, but some schedulers (like with non-zero WarmupSteps) may require per-step updates.
                    optimizer.UpdateLearningRate(step, epoch, epochs);

                    if (allSteps > 1 && consoleOutputMode > ConsoleOutputMode.Disable)
                    {
                        string stepInfo = $"Step {step}/{allSteps}/{epoch}/{epochs}...";
                        string speedAndEtaInfo = string.Empty;

                        if (stepsPerSecond is not null)
                        {
                            long currentTimeInMlliseconds = trainWatch.ElapsedMilliseconds;

                            // Update the speed and ETA every 500 milliseconds
                            if (currentTimeInMlliseconds - calculatedOn >= 500)
                            {
                                long stepsDone = (epoch - 1) * allSteps + step;
                                long remainingSteps = allStepsInTraining - stepsDone;

                                long milisecondsPerStep = currentTimeInMlliseconds / stepsDone;
                                long remainingMiliseconds = remainingSteps * milisecondsPerStep;

                                eta = TimeSpan.FromMilliseconds(remainingMiliseconds);
                                calculatedOn = currentTimeInMlliseconds;
                                speedAndEtaInfo = $" {stepsPerSecond.Value:F2} steps/s - {eta.Hours}h {eta.Minutes}m {eta.Seconds}s left   ";
                            }
                        }

                        Write(stepInfo + speedAndEtaInfo + "\r");
                    }

                    trainLoss = (trainLoss ?? 0) + model.TrainBatch(xBatch, yBatch);
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
                    float testLoss = model.LossFunction.Forward(testPredictions, yTest!);

                    if (consoleOutputMode > ConsoleOutputMode.Disable)
                        WriteLine($"Test loss: {testLoss}");
                    logger?.LogInformation("Test loss: {testLoss} for epoch {epoch}.", testLoss, epoch);

                    if (evalFunction is not null)
                    {
                        float evalValue = evalFunction(model, xTest!, yTest!, testPredictions);

                        if (consoleOutputMode > ConsoleOutputMode.Disable)
                            WriteLine($"Eval: {evalValue:P2}");
                        logger?.LogInformation("Eval: {evalValue:P2} for epoch {epoch}.", evalValue, epoch);
                    }

                    if (testLoss < _bestLoss)
                    {
                        _bestLoss = testLoss;
                        string fileName = $"best_model_{epoch}.json";
                        model.SaveParams(fileName, $"Best model at epoch {epoch} with loss {testLoss:F5}");
                        if(consoleOutputMode > ConsoleOutputMode.Disable)
                            WriteLine($"New best loss: {_bestLoss:F5}. Params saved at {fileName}.");
                        logger?.LogInformation("New best loss: {bestLoss} at epoch {epoch}. Params saved at {fileName}.", _bestLoss, epoch, fileName);
                    }
                    else if (earlyStop)
                    {
                        if (model.HasCheckpoint())
                        {
                            model.RestoreCheckpoint();
                            logger?.LogInformation("Checkpoint restored.");
                        }

                        if (consoleOutputMode > ConsoleOutputMode.Disable)
                            WriteLine($"Early stopping, loss {testLoss} is greater than {_bestLoss}");
                        logger?.LogInformation("Early stopping. Loss {loss} is greater than {bestLoss}.", testLoss, _bestLoss);

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
                WriteLine($"\nFit finished in {elapsedSeconds:F2} s. using {OperationBackend.CurrentType}.");
                WriteLine($"{paramCount:n0} parameters trained.");
                ForegroundColor = ConsoleColor.Yellow;
                TPrediction testPredictions = model.Forward(xTest!, true);
                float testLoss = model.LossFunction.Forward(testPredictions, yTest!);
                WriteLine($"\nLoss on test data: {testLoss:F5}");
                if (evalFunction is not null)
                {
                    float evalValue = evalFunction(model, xTest!, yTest!, testPredictions);
                    WriteLine($"Eval on test data: {evalValue:P2}");
                }
                ResetColor();
                WriteLine();
            }

            if (operationBackendTimingEnabled)
            {
                WriteLine();
                
                string timingReport = OperationBackend.GetStatistics();
                logger?.LogInformation("Operation backend timing report:\n{timingReport}", timingReport);
                if (consoleOutputMode > ConsoleOutputMode.Disable)
                {
                    ForegroundColor = ConsoleColor.Green;
                    WriteLine("Operation backend timing report:");
                    WriteLine(timingReport);
                    ResetColor();
                }
            }

            logger?.LogInformation("===== End Log =====");
            logger?.LogInformation(string.Empty);

        }
        catch (Exception ex)
        {
            logger?.LogError(ex, "An error occurred during training: {message}", ex.Message);
            throw;
        }

        List<string> DescribeFit()
        {
            List<string> res = [];
            res.Add($"Fit (epochs={epochs}, batchSize={batchSize}, evalEveryEpochs={evalEveryEpochs}, logEveryEpochs={logEveryEpochs})");
            res.AddRange(Describe(Constants.Indentation));
            return res;
        }
    }

    public List<string> Describe(int indentation)
    {
        string indent = new(' ', indentation);
        string newIndent = new(' ', indentation + Constants.Indentation);
        List<string> res = [];
        res.Add($"{indent}Trainer");
        res.Add($"{newIndent}Memo: \"{Memo}\"");
        res.Add($"{newIndent}Random: {random}");
        res.Add($"{newIndent}Optimizer: {optimizer}");
        res.Add($"{newIndent}Operation backend: {OperationBackend.CurrentType}");
        res.Add($"{newIndent}Timing enabled: {OperationBackend.StatisticsEnabled}");
        res.AddRange(model.Describe(indentation + Constants.Indentation));
        return res;
    }
}
