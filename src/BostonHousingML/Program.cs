// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

bool running = true;
Console.OutputEncoding = System.Text.Encoding.UTF8;

while (running)
{
    Console.WriteLine("Select a routine to run:");
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
    // 1. Set the hyperparameters for the Boston housing dataset regression task

    const float LearningRate = 0.0005f;
    const int Iterations = 46_000;
    const int PrintEvery = 2_000;

    // 2. Load Boston housing dataset and standardize it (scale features to have mean 0 and stddev 1)

    float[,] bostonData = LoadCsv("..\\..\\..\\..\\..\\data\\Boston\\BostonHousing.csv");

    // Number of samples and coefficients
    int n = bostonData.GetLength(0);

    // Number of independent variables
    int numCoefficients = bostonData.GetLength(1) - 1;

    // Standardize each feature column (mean = 0, stddev = 1) except the target variable (last column)
    for (int j = 0; j < numCoefficients; j++)
    {
        float mean = 0f, std = 0f;
        for (int i = 0; i < n; i++)
        {
            mean += bostonData[i, j];
        }
        mean /= n;
        for (int i = 0; i < n; i++)
        {
            std += MathF.Pow(bostonData[i, j] - mean, 2);
        }
        std = MathF.Sqrt(std / n);
        for (int i = 0; i < n; i++)
        {
            bostonData[i, j] = (bostonData[i, j] - mean) / std;
        }
    }

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
    // Placeholder for FirstNeuralNetwork implementation
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
