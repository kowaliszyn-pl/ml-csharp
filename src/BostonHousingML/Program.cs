// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

bool running = true;
Console.OutputEncoding = System.Text.Encoding.UTF8;

while (running)
{
    Console.WriteLine("Select a routine to run (Boston housing dataset):");
    Console.WriteLine("1. Multiple Linear Regression");
    Console.WriteLine("2. First Neural Network");
    Console.WriteLine("Other: Exit");
    Console.WriteLine();
    Console.Write("Enter your choice: ");

    string? choice = Console.ReadLine();
    Console.WriteLine();

    switch (choice)
    {
        case "1":
            MultipleLinearRegression();
            break;
        case "2":
            FirstNeuralNetwork();
            break;

        default:
            Console.WriteLine("Goodbye!");
            running = false;
            break;
    }

    if (running)
    {
        Console.WriteLine("\nPress any key to continue...");
        Console.ReadKey();
        Console.WriteLine();
    }
}

static void MultipleLinearRegression()
{
    // 1. Set the hyperparameters for the regression task

    const float LearningRate = 0.0005f;
    const int Iterations = 48_000;
    const int PrintEvery = 2_000;

    // 2. Load Boston housing dataset and standardize it (scale features to have mean 0 and stddev 1)

    float[,] bostonData = LoadCsv("..\\..\\..\\..\\..\\data\\Boston\\BostonHousing.csv");

    // Number of samples and coefficients
    int n = bostonData.GetLength(0);

    // Number of independent variables
    int numCoefficients = bostonData.GetLength(1) - 1;

    // Standardize each feature column (mean = 0, stddev = 1) except the target variable (last column)
    // Note: the upper bound in Range is exclusive, so we use numCoefficients to exclude the last column
    bostonData.Standardize(0..numCoefficients); 

    // 3. Convert data to matrices with bias term

    float[,] XAnd1 = new float[n, numCoefficients + 1]; // +1 for bias term
    float[,] Y = new float[n, 1];

    // Prepare feature matrix XAnd1 with bias term and target vector Y
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < numCoefficients; j++)
        {
            XAnd1[i, j] = bostonData[i, j];
        }

        // Add bias term
        XAnd1[i, numCoefficients] = 1;

        // Target values
        Y[i, 0] = bostonData[i, numCoefficients];
    }

    // 4. Initialize model parameters

    // These are the coefficients for our independent variables and the bias term
    // initialized to zero
    float[,] AB = new float[numCoefficients + 1, 1];

    // 5. Training loop

    var XAnd1T = XAnd1.Transpose();
    var negativeTwoOverN = -2.0f / n;
    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Prediction and error calculation

        // Make predictions for all samples at once: predictions = XAnd1 * AB
        float[,] predictions = XAnd1.MultiplyDot(AB);

        // Calculate errors for all samples: errors = Y - predictions
        float[,] errors = Y.Subtract(predictions);

        // Calculate gradient for coefficients 'AB': ∂MSE/∂AB = -2/n * XAnd1^T * errors
        float[,] deltaAB = XAnd1T.MultiplyDot(errors).Multiply(negativeTwoOverN);

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

            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5} | a1: {AB[0, 0],8:F4} | a2: {AB[1, 0],8:F4} | a3: {AB[2, 0],8:F4} | ... | b: {AB[numCoefficients, 0],8:F4}");
        }
    }

    // 6. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Matrices with Bias on Boston Data) ---");
    Console.WriteLine("Learned parameters:");

    for (int i = 0; i < numCoefficients; i++)
    {
        Console.WriteLine($" a{i + 1,-2} = {AB[i, 0],8:F4}");
    }
    Console.WriteLine($" b   = {AB[numCoefficients, 0],8:F4}");

    Console.WriteLine();
    Console.WriteLine("Sample predictions vs actual values:");
    Console.WriteLine();
    Console.WriteLine($"{"Sample No",14}{"Predicted",14}{"Actual",14}");
    Console.WriteLine();

    int[] showSamples = { 0, 1, 2, 3, 4, n - 1 };

    for (int i = 0; i < showSamples.Length; i++)
    {
        int sampleIndex = showSamples[i];
        float predicted = XAnd1.GetRowAs2D(sampleIndex).MultiplyDot(AB)[0, 0];
        float actual = Y[sampleIndex, 0];
        Console.WriteLine(
            $"{sampleIndex + 1,14}" +
            $"{predicted,14:F4}" +
            $"{actual,14:F4}"
        );
    }
}

static void FirstNeuralNetwork()
{
    // 1. Set the hyperparameters for the neural network

    const float LearningRate = 0.0005f;
    const int Iterations = 48_000;
    const int PrintEvery = 2_000;
    const int HiddenLayerSize = 4;

    // 2. Load Boston housing dataset and standardize it (scale features to have mean 0 and stddev 1)

    float[,] bostonData = LoadCsv("..\\..\\..\\..\\..\\data\\Boston\\BostonHousing.csv");

    // Number of samples and coefficients
    int n = bostonData.GetLength(0);

    // Number of independent variables
    int numCoefficients = bostonData.GetLength(1) - 1;

    // Standardize each feature column (mean = 0, stddev = 1) except the target variable (last column)
    // Note: the upper bound in Range is exclusive, so we use numCoefficients to exclude the last column
    bostonData.Standardize(0..numCoefficients);

    // 3. Convert data to matrices

    float[,] X = new float[n, numCoefficients];
    float[,] Y = new float[n, 1];

    // Prepare feature matrix X and target vector Y
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < numCoefficients; j++)
        {
            X[i, j] = bostonData[i, j];
        }

        // Target values
        Y[i, 0] = bostonData[i, numCoefficients];
    }

    // 4. Initialize model parameters

    int seed = 42;

    // Weights and biases for the first layer
    float[,] W1 = new float[numCoefficients, HiddenLayerSize];
    W1.RandomInPlace(seed);
    float[] B1 = new float[HiddenLayerSize];

    // Weights and biases for the second layer
    float[,] W2 = new float[HiddenLayerSize, 1];
    W2.RandomInPlace(seed + 1);
    float b2 = 0;

    // 5. Training loop

    var XT = X.Transpose();
    var negativeTwoOverN = -2.0f / n;
    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Model structure: X → [W1, B1] → sigmoid → [W2, b2] → output
        // Forward (prediction and error calculation)

        // The first layer
        float[,] M1 = X.MultiplyDot(W1);
        float[,] N1 = M1.AddRow(B1);
        float[,] O1 = N1.Sigmoid();

        // The second layer
        float[,] M2 = O1.MultiplyDot(W2);
        float[,] predictions = M2.Add(b2);

        // Calculate errors for all samples: errors = Y - predictions
        float[,] errors = Y.Subtract(predictions);

        float meanSquaredError = errors.Power(2).Mean();

        // Back (gradient calculation and parameters update)

        // The first layer
        float[,] dLdP = errors.Multiply(negativeTwoOverN);
        float[,] dPdM2 = M2.AsOnes();
        float[,] dLdM2 = dLdP.MultiplyElementwise(dPdM2);
        float dPdBias2 = 1;
        float dLBias2 = dLdP.Multiply(dPdBias2).Mean();
        float[,] dM2dW2 = O1.Transpose();
        float[,] dLdW2 = dM2dW2.MultiplyDot(dLdP);

        // The second layer
        float[,] dM2dO1 = W2.Transpose();
        float[,] dLdO1 = dLdM2.MultiplyDot(dM2dO1);
        float[,] dO1dN1 = N1.SigmoidDerivative();
        float[,] dLdN1 = dLdO1.MultiplyElementwise(dO1dN1);
        float[] dN1dBias1 = B1.AsOnes();
        float[,] dN1dM1 = M1.AsOnes();
        float[] dLdBias1 = dN1dBias1.MultiplyElementwise(dLdN1).MeanByColumn();
        float[,] dLdM1 = dLdN1.MultiplyElementwise(dN1dM1);
        float[,] dM1dW1 = XT;
        float[,] dLdW1 = dM1dW1.MultiplyDot(dLdM1);

        // Update parameters
        W1 = W1.Subtract(dLdW1.Multiply(LearningRate));
        W2 = W2.Subtract(dLdW2.Multiply(LearningRate));
        B1 = B1.Subtract(dLdBias1.Multiply(LearningRate));
        b2 -= dLBias2 * LearningRate;

        if (iteration % PrintEvery == 0)
        {
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
    int[] showSamples = { 0, 1, 2, 3, 4, n - 1 };
    for (int i = 0; i < showSamples.Length; i++)
    {
        int sampleIndex = showSamples[i];
        // Forward pass for the sample
        float[,] M1 = X.GetRowAs2D(sampleIndex).MultiplyDot(W1);
        float[,] N1 = M1.AddRow(B1);
        float[,] O1 = N1.Sigmoid();
        float[,] M2 = O1.MultiplyDot(W2);
        float[,] prediction = M2.Add(b2);
        float predicted = prediction[0, 0];
        float actual = Y[sampleIndex, 0];
        Console.WriteLine(
            $"{sampleIndex + 1,14}" +
            $"{predicted,14:F4}" +
            $"{actual,14:F4}"
        );
    }
}

static float[,] LoadCsv(string filePath)
{
    string[] lines = [.. File.ReadAllLines(filePath).Skip(1)];
    int rows = lines.Length;
    int cols = lines[0].Split(',').Length;
    float[,] matrix = new float[rows, cols];
    for (int i = 0; i < rows; i++)
    {
        string[] values = lines[i].Split(',');
        for (int j = 0; j < cols; j++)
        {
            string value = values[j].Trim('"');
            matrix[i, j] = float.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
        }
    }
    return matrix;
}
