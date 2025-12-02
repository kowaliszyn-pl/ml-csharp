// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core;

using static NeuralNetworks.Core.ArrayUtils;

bool running = true;
Console.OutputEncoding = System.Text.Encoding.UTF8;

while (running)
{
    Console.WriteLine("Select a routine to run (Boston housing dataset):");
    Console.WriteLine("1. Multiple Linear Regression");
    Console.WriteLine("2. First Neural Network");
    Console.WriteLine("3. First Neural Network (Simplified calculations)");
    Console.WriteLine("Other: Exit");
    Console.WriteLine();
    Console.Write("Enter your choice: ");

    string? choice = Console.ReadLine();
    Console.WriteLine();

    Stopwatch stopwatch = Stopwatch.StartNew();
    switch (choice)
    {
        case "1":
            MultipleLinearRegression();
            break;
        case "2":
            FirstNeuralNetwork();
            break;
        case "3":
            FirstNeuralNetworkSimplified();
            break;

        default:
            Console.WriteLine("Goodbye!");
            running = false;
            break;
    }

    if (running)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"Elapsed time: ~{stopwatch.Elapsed.TotalSeconds:F2} seconds.");
        Console.ResetColor();

        Console.WriteLine("\nPress any key to continue...");
        Console.ReadKey();
        Console.WriteLine();
    }
}

// Define hyperparameters for both routines

const float LearningRate = 0.0005f;
const int Iterations = 48_000;
const int PrintEvery = 2_000;
const float TestSplitRatio = 0.7f;
const int RandomSeed = 251113;

static void MultipleLinearRegression()
{
    // 1. Get data

    (float[,] trainData, float[,] testData) = GetData();

    // 2. Copy trainData and testData to matrices with bias term

    int inputFeatureCount = trainData.GetLength(1) - 1;
    int nTrain = trainData.GetLength(0);
    int nTest = testData.GetLength(0);

    float[,] XTrainAnd1 = new float[nTrain, inputFeatureCount + 1];
    float[,] YTrain = new float[nTrain, 1];

    float[,] XTestAnd1 = new float[nTest, inputFeatureCount + 1];
    float[,] YTest = new float[nTest, 1];

    // Prepare feature matrix XTrainAnd1 with bias term and target vector YTrain
    for (int i = 0; i < nTrain; i++)
    {
        for (int j = 0; j < inputFeatureCount; j++)
        {
            XTrainAnd1[i, j] = trainData[i, j];
        }

        // Add bias term
        XTrainAnd1[i, inputFeatureCount] = 1;

        // Target values
        YTrain[i, 0] = trainData[i, inputFeatureCount];
    }

    // Prepare feature matrix XTestAnd1 with bias term and target vector YTest
    for (int i = 0; i < nTest; i++)
    {
        for (int j = 0; j < inputFeatureCount; j++)
        {
            XTestAnd1[i, j] = testData[i, j];
        }

        // Add bias term
        XTestAnd1[i, inputFeatureCount] = 1;

        // Target values
        YTest[i, 0] = testData[i, inputFeatureCount];
    }

    // 3. Initialize model parameters

    // Coefficients for our independent variables and the bias term initialized to zero
    float[,] AB = new float[inputFeatureCount + 1, 1];

    // 4. Training loop

    float[,] XTrainAnd1T = XTrainAnd1.Transpose();
    float negativeTwoOverN = -2.0f / nTrain;
    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Prediction and error calculation

        // Make predictions for all samples at once: predictions = XTrainAnd1 * AB
        float[,] predictions = XTrainAnd1.MultiplyDot(AB);

        // Calculate errors for all samples: errors = YTrain - predictions
        float[,] errors = YTrain.Subtract(predictions);

        // Calculate gradient for coefficients 'AB': ∂MSE/∂AB = -2/n * XTrainAnd1^T * errors
        float[,] deltaAB = XTrainAnd1T.MultiplyDot(errors).Multiply(negativeTwoOverN);

        // Update regression parameters using gradient descent
        AB = AB.Subtract(deltaAB.Multiply(LearningRate));

        if (iteration % PrintEvery == 0)
        {
            // Calculate the Mean Squared Error loss: MSE = mean(errors^2)
            float meanSquaredError = errors.Power(2).Mean();

            if (float.IsNaN(meanSquaredError))
            {
                Console.WriteLine($"NaN detected at iteration {iteration}");
                break;
            }

            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5} | a1: {AB[0, 0],8:F4} | a2: {AB[1, 0],8:F4} | a3: {AB[2, 0],8:F4} | ... | b: {AB[inputFeatureCount, 0],8:F4}");
        }
    }

    // 5. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Matrices with Bias on Boston Data) ---");
    Console.WriteLine("Learned parameters:");

    for (int i = 0; i < inputFeatureCount; i++)
    {
        Console.WriteLine($" a{i + 1,-2} = {AB[i, 0],8:F4}");
    }
    Console.WriteLine($" b   = {AB[inputFeatureCount, 0],8:F4}");

    Console.WriteLine();
    Console.WriteLine("Sample predictions vs actual values:");
    Console.WriteLine();
    Console.WriteLine($"{"Sample No",14}{"Predicted",14}{"Actual",14}");
    Console.WriteLine();

    // Show predictions for the test set

    int[] showTestSamples = { 0, 1, 2, nTest - 3, nTest - 2, nTest - 1 };
    float[,] testPredictions = XTestAnd1.MultiplyDot(AB);

    for (int i = 0; i < showTestSamples.Length; i++)
    {
        int testSampleIndex = showTestSamples[i];
        float predicted = testPredictions[testSampleIndex, 0];
        float actual = YTest[testSampleIndex, 0];
        Console.WriteLine(
            $"{testSampleIndex + 1,14}" +
            $"{predicted,14:F4}" +
            $"{actual,14:F4}"
        );
    }

    // Show MSE for test data

    float[,] testErrors = YTest.Subtract(testPredictions);
    float testMeanSquaredError = testErrors.Power(2).Mean();
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"\nMSE on test data: {testMeanSquaredError:F5}");
    Console.ResetColor();
}

static void FirstNeuralNetwork()
{
    // 1. Set the hyperparameters for the neural network

    const int HiddenLayerSize = 4;

    // 2. Get data

    (float[,] trainData, float[,] testData) = GetData();

    // 3. Copy trainData and testData to matrices

    int inputFeatureCount = trainData.GetLength(1) - 1;
    int nTrain = trainData.GetLength(0);
    int nTest = testData.GetLength(0);

    float[,] XTrain = new float[nTrain, inputFeatureCount];
    float[,] YTrain = new float[nTrain, 1];

    float[,] XTest = new float[nTest, inputFeatureCount];
    float[,] YTest = new float[nTest, 1];

    // Prepare feature matrix XTrain and target vector YTrain
    for (int i = 0; i < nTrain; i++)
    {
        for (int j = 0; j < inputFeatureCount; j++)
        {
            XTrain[i, j] = trainData[i, j];
        }

        // Target values
        YTrain[i, 0] = trainData[i, inputFeatureCount];
    }

    // Prepare feature matrix XTest and target vector YTest
    for (int i = 0; i < nTest; i++)
    {
        for (int j = 0; j < inputFeatureCount; j++)
        {
            XTest[i, j] = testData[i, j];
        }
        // Target values
        YTest[i, 0] = testData[i, inputFeatureCount];
    }

    // 4. Initialize model parameters

    // Weights and biases for the first layer
    float[,] W1 = new float[inputFeatureCount, HiddenLayerSize];
    W1.RandomInPlace(RandomSeed);
    float[] B1 = new float[HiddenLayerSize];

    // Weights and biases for the second layer
    float[,] W2 = new float[HiddenLayerSize, 1];

    // We use RandomSeed + 1 because we want different random values than for W1  
    W2.RandomInPlace(RandomSeed + 1);
    float b2 = 0;

    // 5. Training loop

    float[,] XTrainT = XTrain.Transpose();
    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Model structure: XTrain → [W1, B1] → sigmoid → [W2, b2] → output

        // 5.1. Forward (prediction and error calculation)

        /* 
           W1 - weights for the first layer [ inputSize (no of columns/attributes of X) x hiddenSize ]
           W2 - weights for the second layer [ hiddenSize x 1 ]
           B1 - bias for the first layer (for every neuron in the first layer)
           b2 - bias for the second layer (there is only one neuron in the second layer)
           M1 - input multiplied by W1
           N1 - input multiplied by W1 plus B1
           O1 - result of the activation function applied to (input multiplied by W1 plus B1)
           M2 - result of O1 (result of the activation function from the first layer) multiplied by W2
           predictions - (M2 + b2)
           errors - subtract predictions from YTrain
           meanSquaredError - MSE, mean of errors squared
        */

        // == The first layer (hidden) ==
        float[,] M1 = XTrain.MultiplyDot(W1);
        float[,] N1 = M1.AddRow(B1);
        // Apply sigmoid activation function, so we can get O1 - outputs of the first layer
        float[,] O1 = N1.Sigmoid();

        // == The second layer (output) ==
        float[,] M2 = O1.MultiplyDot(W2);
        float[,] predictions = M2.Add(b2);

        // Calculate errors for all samples: errors = YTrain - predictions
        float[,] errors = YTrain.Subtract(predictions);

        // 5.2. Back (gradient calculation and parameters update). We do all calculations in backward order.

        // == The second layer (output) ==

        // [nTrain, 1]
        float[,] dLdP = errors.Multiply(-2.0f / nTrain);

        // [nTrain, 1]
        float[,] dPdM2 = M2.AsOnes();

        // [nTrain, 1]
        float[,] dLdM2 = dLdP.MultiplyElementwise(dPdM2);

        float dPdBias2 = 1;

        // mean([nTrain, 1]) -> scalar
        float dLdBias2 = dLdP.Multiply(dPdBias2).Sum();

        // [HiddenLayerSize, nTrain]
        float[,] dM2dW2 = O1.Transpose();

        // [HiddenLayerSize, 1]
        float[,] dLdW2 = dM2dW2.MultiplyDot(dLdP);

        // == The first layer (hidden) == 

        // [1, HiddenLayerSize]
        float[,] dM2dO1 = W2.Transpose();

        // [nTrain, HiddenLayerSize]
        float[,] dLdO1 = dLdM2.MultiplyDot(dM2dO1);

        // [nTrain, HiddenLayerSize]
        float[,] dO1dN1 = N1.SigmoidDerivative();

        // [nTrain, HiddenLayerSize]
        float[,] dLdN1 = dLdO1.MultiplyElementwise(dO1dN1);

        // [HiddenLayerSize]
        float[] dN1dBias1 = B1.AsOnes();

        // [nTrain, HiddenLayerSize]
        float[,] dN1dM1 = M1.AsOnes();

        // [HiddenLayerSize]
        float[] dLdBias1 = dN1dBias1.MultiplyElementwise(dLdN1).SumByColumns();

        // [nTrain, HiddenLayerSize]
        float[,] dLdM1 = dLdN1.MultiplyElementwise(dN1dM1);

        // [inputFeatureCount, nTrain]
        float[,] dM1dW1 = XTrainT;

        // [inputFeatureCount, HiddenLayerSize]
        float[,] dLdW1 = dM1dW1.MultiplyDot(dLdM1);

        // Update parameters
        W1 = W1.Subtract(dLdW1.Multiply(LearningRate));
        W2 = W2.Subtract(dLdW2.Multiply(LearningRate));
        B1 = B1.Subtract(dLdBias1.Multiply(LearningRate));
        b2 -= dLdBias2 * LearningRate;

        if (iteration % PrintEvery == 0)
        {
            float meanSquaredError = errors.Power(2).Mean();

            // Calculate the Mean Squared Error loss: MSE = mean(errors^2)
            if (float.IsNaN(meanSquaredError))
            {
                Console.WriteLine($"NaN detected at iteration {iteration}");
                break;
            }

            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5}");
        }
    }

    // 6. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Neural Network on Boston Data) ---");
    Console.WriteLine("Learned parameters:");
    Console.WriteLine("Weights for the first layer (W1):");
    for (int i = 0; i < W1.GetLength(0); i++)
    {
        for (int j = 0; j < W1.GetLength(1); j++)
        {
            Console.Write($"{W1[i, j],8:F4} ");
        }
        Console.WriteLine();
    }
    Console.WriteLine("Biases for the first layer (B1):");
    for (int j = 0; j < B1.Length; j++)
    {
        Console.WriteLine($" B1[{j}] = {B1[j],8:F4}");
    }
    Console.WriteLine("Weights for the second layer (W2):");
    for (int i = 0; i < W2.GetLength(0); i++)
    {
        for (int j = 0; j < W2.GetLength(1); j++)
        {
            Console.Write($"{W2[i, j],8:F4} ");
        }
        Console.WriteLine();
    }
    Console.WriteLine($"Bias for the second layer (b2): {b2,8:F4}");
    Console.WriteLine();
    Console.WriteLine("Sample predictions vs actual values:");
    Console.WriteLine();
    Console.WriteLine($"{"Sample No",14}{"Predicted",14}{"Actual",14}");
    Console.WriteLine();

    // Show predictions for the test set

    int[] showTestSamples = { 0, 1, 2, nTest - 3, nTest - 2, nTest - 1 };

    // Do a forward pass for all test samples at once

    // The first layer (hidden)
    float[,] M1Test = XTest.MultiplyDot(W1);
    float[,] N1Test = M1Test.AddRow(B1);
    // Apply sigmoid activation function, so we can get O1 - outputs of the first layer
    float[,] O1Test = N1Test.Sigmoid();
    // The second layer (output)
    float[,] M2Test = O1Test.MultiplyDot(W2);
    float[,] testPredictions = M2Test.Add(b2);

    for (int i = 0; i < showTestSamples.Length; i++)
    {
        int testSampleIndex = showTestSamples[i];
        float predicted = testPredictions[testSampleIndex, 0];
        float actual = YTest[testSampleIndex, 0];
        Console.WriteLine(
            $"{testSampleIndex + 1,14}" +
            $"{predicted,14:F4}" +
            $"{actual,14:F4}"
        );
    }

    // Show MSE for test data

    float[,] testErrors = YTest.Subtract(testPredictions);
    float testMeanSquaredError = testErrors.Power(2).Mean();
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"\nMSE on test data: {testMeanSquaredError:F5}");
    Console.ResetColor();
}

static void FirstNeuralNetworkSimplified()
{
    // 1. Set the hyperparameters for the neural network

    const int HiddenLayerSize = 4;

    // 2. Get data

    (float[,] trainData, float[,] testData) = GetData();

    // 3. Copy trainData and testData to matrices

    int inputFeatureCount = trainData.GetLength(1) - 1;
    int nTrain = trainData.GetLength(0);
    int nTest = testData.GetLength(0);

    float[,] XTrain = new float[nTrain, inputFeatureCount];
    float[,] YTrain = new float[nTrain, 1];

    float[,] XTest = new float[nTest, inputFeatureCount];
    float[,] YTest = new float[nTest, 1];

    // Prepare feature matrix XTrain and target vector YTrain
    for (int i = 0; i < nTrain; i++)
    {
        for (int j = 0; j < inputFeatureCount; j++)
        {
            XTrain[i, j] = trainData[i, j];
        }

        // Target values
        YTrain[i, 0] = trainData[i, inputFeatureCount];
    }

    // Prepare feature matrix XTest and target vector YTest
    for (int i = 0; i < nTest; i++)
    {
        for (int j = 0; j < inputFeatureCount; j++)
        {
            XTest[i, j] = testData[i, j];
        }
        // Target values
        YTest[i, 0] = testData[i, inputFeatureCount];
    }

    // 4. Initialize model parameters

    // Weights and biases for the first layer
    float[,] W1 = new float[inputFeatureCount, HiddenLayerSize];
    W1.RandomInPlace(RandomSeed);
    float[] B1 = new float[HiddenLayerSize];

    // Weights and biases for the second layer
    float[,] W2 = new float[HiddenLayerSize, 1];

    // We use RandomSeed + 1 because we want different random values than for W1  
    W2.RandomInPlace(RandomSeed + 1);
    float b2 = 0;

    // 5. Training loop

    float[,] XTrainT = XTrain.Transpose();
    float negativeTwoOverN = -2.0f / nTrain;
    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Model structure: XTrain → [W1, B1] → sigmoid → [W2, b2] → output

        // 5.1. Forward (prediction and error calculation)

        // The first layer (hidden)
        float[,] M1 = XTrain.MultiplyDot(W1);
        float[,] N1 = M1.AddRow(B1);
        // Apply sigmoid activation function, so we can get O1 - outputs of the first layer
        float[,] O1 = N1.Sigmoid();

        // The second layer (output)
        float[,] M2 = O1.MultiplyDot(W2);
        float[,] predictions = M2.Add(b2);

        // Calculate errors for all samples: errors = YTrain - predictions
        float[,] errors = YTrain.Subtract(predictions);

        // 5.2. Back (gradient calculation and parameters update). We do all calculations in backward order. This time a little bit simplified.

        // The second layer (output)
        float[,] dLdP = errors.Multiply(negativeTwoOverN);
        float dLdBias2 = dLdP.Sum();
        float[,] dLdW2 = O1.Transpose().MultiplyDot(dLdP);

        // The first layer (hidden)
        float[,] dLdO1 = dLdP.MultiplyDot(W2.Transpose());
        float[,] dLdN1 = dLdO1.MultiplyElementwise(N1.SigmoidDerivative());
        float[] dLdBias1 = dLdN1.SumByColumns();
        float[,] dLdW1 = XTrainT.MultiplyDot(dLdN1);

        // Update parameters
        W1 = W1.Subtract(dLdW1.Multiply(LearningRate));
        W2 = W2.Subtract(dLdW2.Multiply(LearningRate));
        B1 = B1.Subtract(dLdBias1.Multiply(LearningRate));
        b2 -= dLdBias2 * LearningRate;

        if (iteration % PrintEvery == 0)
        {
            float meanSquaredError = errors.Power(2).Mean();

            // Calculate the Mean Squared Error loss: MSE = mean(errors^2)
            if (float.IsNaN(meanSquaredError))
            {
                Console.WriteLine($"NaN detected at iteration {iteration}");
                break;
            }

            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5}");
        }
    }

    // 6. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Simplified Neural Network on Boston Data) ---");
    Console.WriteLine("Learned parameters:");
    Console.WriteLine("Weights for the first layer (W1):");
    for (int i = 0; i < W1.GetLength(0); i++)
    {
        for (int j = 0; j < W1.GetLength(1); j++)
        {
            Console.Write($"{W1[i, j],8:F4} ");
        }
        Console.WriteLine();
    }
    Console.WriteLine("Biases for the first layer (B1):");
    for (int j = 0; j < B1.Length; j++)
    {
        Console.WriteLine($" B1[{j}] = {B1[j],8:F4}");
    }
    Console.WriteLine("Weights for the second layer (W2):");
    for (int i = 0; i < W2.GetLength(0); i++)
    {
        for (int j = 0; j < W2.GetLength(1); j++)
        {
            Console.Write($"{W2[i, j],8:F4} ");
        }
        Console.WriteLine();
    }
    Console.WriteLine($"Bias for the second layer (b2): {b2,8:F4}");
    Console.WriteLine();
    Console.WriteLine("Sample predictions vs actual values:");
    Console.WriteLine();
    Console.WriteLine($"{"Sample No",14}{"Predicted",14}{"Actual",14}");
    Console.WriteLine();

    // Show predictions for the test set

    int[] showTestSamples = { 0, 1, 2, nTest - 3, nTest - 2, nTest - 1 };

    // Do a forward pass for all test samples at once

    // The first layer (hidden)
    float[,] M1Test = XTest.MultiplyDot(W1);
    float[,] N1Test = M1Test.AddRow(B1);
    // Apply sigmoid activation function, so we can get O1 - outputs of the first layer
    float[,] O1Test = N1Test.Sigmoid();
    // The second layer (output)
    float[,] M2Test = O1Test.MultiplyDot(W2);
    float[,] testPredictions = M2Test.Add(b2);

    for (int i = 0; i < showTestSamples.Length; i++)
    {
        int testSampleIndex = showTestSamples[i];
        float predicted = testPredictions[testSampleIndex, 0];
        float actual = YTest[testSampleIndex, 0];
        Console.WriteLine(
            $"{testSampleIndex + 1,14}" +
            $"{predicted,14:F4}" +
            $"{actual,14:F4}"
        );
    }

    // Show MSE for test data

    float[,] testErrors = YTest.Subtract(testPredictions);
    float testMeanSquaredError = testErrors.Power(2).Mean();
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"\nMSE on test data: {testMeanSquaredError:F5}");
    Console.ResetColor();
}

static (float[,] TrainData, float[,] TestData) GetData()
{
    float[,] bostonData = LoadCsv("..\\..\\..\\..\\..\\data\\Boston\\BostonHousing.csv", 1);

    // Number of independent variables
    int inputFeatureCount = bostonData.GetLength(1) - 1;

    // Standardize each feature column (mean = 0, stddev = 1) except the target variable (last column)
    // Note: the upper bound in Range is exclusive, so we use inputFeatureCount to exclude the last column
    bostonData.Standardize(0..inputFeatureCount);

    // Permute the data randomly
    bostonData.PermuteInPlace(RandomSeed);

    // Return train and test data split by ratio
    return bostonData.SplitRowsByRatio(TestSplitRatio);
}

//static float[,] LoadCsv(string filePath)
//{
//    string[] lines = [.. File.ReadAllLines(filePath).Skip(1)];
//    int rows = lines.Length;
//    int cols = lines[0].Split(',').Length;
//    float[,] matrix = new float[rows, cols];
//    for (int i = 0; i < rows; i++)
//    {
//        string[] values = lines[i].Split(',');
//        for (int j = 0; j < cols; j++)
//        {
//            string value = values[j].Trim('"');
//            matrix[i, j] = float.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
//        }
//    }
//    return matrix;
//}
