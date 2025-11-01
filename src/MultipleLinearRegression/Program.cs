// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

// Set the parameters for the model

const float LearningRate = 0.0005f;
const int Iterations = 35_000;
const int PrintEvery = 1_000;

// Prepare training data
// Each inner array represents a sample: [x1, x2, x3, y]
// We will be trying to find the relationship: y = 2*x1 + 3*x2 - 1*x3 + 5

float[][] data = [
    [1, 2, 1, 12], // y = 2*1 + 3*2 - 1*1 + 5 = 12
    [2, 1, 2, 10], // etc.
    [3, 3, 1, 19],
    [4, 2, 3, 16],
    [1, 4, 2, 17]
];

bool running = true;
Console.OutputEncoding = System.Text.Encoding.UTF8;

while (running)
{
    Console.WriteLine("Select a routine to run:");
    Console.WriteLine("1. Variables");
    Console.WriteLine("2. Tables");
    Console.WriteLine("3. Matrices");
    Console.WriteLine("4. Matrices with bias");
    Console.WriteLine("5. Matrices with bias on the Boston data");
    Console.WriteLine("Other: Exit");
    Console.WriteLine();
    Console.Write("Enter your choice: ");

    string? choice = Console.ReadLine();
    Console.WriteLine();

    switch (choice)
    {
        case "1":
            Variables();
            break;
        case "2":
            Tables();
            break;
        case "3":
            Matrices();
            break;
        case "4":
            MatricesWithBias();
            break;
        case "5":
            MatricesWithBiasBoston();
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

void Variables()
{
    // 1. Initialize model parameters

    // These are the coefficients for our independent variables and the bias term
    float a1 = 0, a2 = 0, a3 = 0; // Parameters for x1, x2, x3
    float b = 0;

    // Number of samples
    int n = data.Length;

    // 2. Training loop

    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Initialize accumulators for errors and gradients for this iteration
        float sumSquaredError = 0;
        float sumErrorForA1 = 0;
        float sumErrorForA2 = 0;
        float sumErrorForA3 = 0;
        float sumErrorForB = 0;

        foreach (float[] sample in data)
        {
            // Get the independent variables (features) and the dependent variable (target)
            float x1 = sample[0];
            float x2 = sample[1];
            float x3 = sample[2];
            float y = sample[3];

            // Prediction and error calculation
            float prediction = a1 * x1 + a2 * x2 + a3 * x3 + b;
            float error = y - prediction;

            // Accumulate squared error for MSE calculation
            sumSquaredError += error * error;

            // Accumulate parts needed for gradient calculation
            sumErrorForA1 += error * x1;
            sumErrorForA2 += error * x2;
            sumErrorForA3 += error * x3;
            sumErrorForB += error;
        }

        // Calculate gradients (partial derivatives of MSE)
        // ∂MSE/∂a1 = -2/n * Σ(error * x1)
        float deltaA1 = -2.0f / n * sumErrorForA1;
        float deltaA2 = -2.0f / n * sumErrorForA2;
        float deltaA3 = -2.0f / n * sumErrorForA3;
        float deltaB = -2.0f / n * sumErrorForB;

        // Update regression parameters using gradient descent
        a1 -= LearningRate * deltaA1;
        a2 -= LearningRate * deltaA2;
        a3 -= LearningRate * deltaA3;
        b -= LearningRate * deltaB;

        if (iteration % PrintEvery == 0)
        {
            // MSE
            float meanSquaredError = sumSquaredError / n;

            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5} | a1: {a1,8:F4} | a2: {a2,8:F4} | a3: {a3,8:F4} | b: {b,8:F4}");
        }
    }

    // 3. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Variables) ---");
    Console.WriteLine($"{"Learned parameters:",-20} a1 = {a1,9:F4} | a2 = {a2,9:F4} | a3 = {a3,9:F4} | b = {b,9:F4}");
    Console.WriteLine($"{"Expected parameters:",-20} a1 = {2,9:F4} | a2 = {3,9:F4} | a3 = {-1,9:F4} | b = {5,9:F4}");
}

void Tables()
{
    // 1. Initialize model parameters

    // These are the coefficients for our independent variables and the bias term
    int numCoefficients = data[0].Length - 1; // Number of independent variables (3 in this case)
    float[] a = new float[numCoefficients]; // Corresponds to a1, a2, a3. It's already initialized to 0 at this point.
    float b = 0;

    // Number of samples and coefficients
    int n = data.Length;

    // 2. Training loop

    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Initialize accumulators for errors and gradients for this iteration
        float sumSquaredError = 0;
        float[] sumErrorForA = new float[numCoefficients]; // Accumulator for each coefficient's gradient part
        float sumErrorForB = 0; // Accumulator for the bias's gradient part

        foreach (float[] sample in data)
        {
            // Separate independent variables (features) (x) from the dependent variable (target) (y)
            float[] x = new float[numCoefficients];
            for (int i = 0; i < numCoefficients; i++)
            {
                x[i] = sample[i];
            }
            float y = sample[numCoefficients];

            // Prediction and error calculation
            // prediction = a1*x1 + a2*x2 + a3*x3 + b
            float prediction = b;
            for (int i = 0; i < numCoefficients; i++)
            {
                prediction += a[i] * x[i];
            }
            float error = y - prediction;

            // Accumulate squared error for MSE calculation
            sumSquaredError += error * error;

            // Accumulate parts needed for gradient calculation
            // For each ai, the gradient part is (error * xi)
            for (int i = 0; i < numCoefficients; i++)
            {
                sumErrorForA[i] += error * x[i];
            }
            // For the bias, the gradient part is just the error
            sumErrorForB += error;
        }

        // Calculate gradients (partial derivatives of MSE)
        // ∂MSE/∂ai = -2/n * Σ(error * xi)
        float[] deltaA = new float[numCoefficients];
        for (int i = 0; i < numCoefficients; i++)
        {
            deltaA[i] = -2.0f / n * sumErrorForA[i];
        }

        // ∂MSE/∂b = -2/n * Σ(error)
        float deltaB = -2.0f / n * sumErrorForB;

        // Update regression parameters using gradient descent
        for (int i = 0; i < numCoefficients; i++)
        {
            a[i] -= LearningRate * deltaA[i];
        }
        b -= LearningRate * deltaB;

        if (iteration % PrintEvery == 0)
        {
            // MSE
            float meanSquaredError = sumSquaredError / n;

            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5} | a1: {a[0],8:F4} | a2: {a[1],8:F4} | a3: {a[2],8:F4} | b: {b,8:F4}");
        }
    }

    // 3. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Tables) ---");
    Console.WriteLine($"{"Learned parameters:",-20} a1 = {a[0],9:F4} | a2 = {a[1],9:F4} | a3 = {a[2],9:F4} | b = {b,9:F4}");
    Console.WriteLine($"{"Expected parameters:",-20} a1 = {2,9:F4} | a2 = {3,9:F4} | a3 = {-1,9:F4} | b = {5,9:F4}");
}

void Matrices()
{
    // 1. Convert data to matrices

    // Number of samples and coefficients
    int n = data.Length;
    int numCoefficients = data[0].Length - 1; // Number of independent variables (3 in this case)

    float[,] X = new float[n, numCoefficients];
    float[,] Y = new float[n, 1];

    // Prepare feature matrix X and target vector Y
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < numCoefficients; j++)
        {
            X[i, j] = data[i][j];
        }
        Y[i, 0] = data[i][numCoefficients];
    }

    // 2. Initialize model parameters

    // These are the coefficients for our independent variables and the bias term
    float[,] A = new float[numCoefficients, 1]; // Corresponds to a1, a2, a3. It's already initialized to 0 at this point.
    float b = 0;

    // 3. Training loop

    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Prediction and error calculation

        // Make predictions for all samples at once: predictions = X * a + b
        float[,] predictions = X.MultiplyDot(A).Add(b);

        // Calculate errors for all samples: errors = Y - predictions
        float[,] errors = Y.Subtract(predictions);

        // Calculate gradient for coefficients 'a': ∂MSE/∂a = -2/n * X^T * errors
        // X.Transpose() aligns features with their corresponding errors for the dot product.
        float[,] deltaA = X.Transpose().MultiplyDot(errors).Multiply(-2.0f / n);

        // ∂MSE/∂b = -2/n * sum(errors)
        float deltaB = -2.0f / n * errors.Sum();

        // Update regression parameters using gradient descent
        A = A.Subtract(deltaA.Multiply(LearningRate));
        b -= LearningRate * deltaB;

        if (iteration % PrintEvery == 0)
        {
            // Calculate the Mean Squared Error loss: MSE = mean(errors^2)
            float meanSquaredError = errors.Power(2).Mean();

            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5} | a1: {A[0, 0],8:F4} | a2: {A[1, 0],8:F4} | a3: {A[2, 0],8:F4} | b: {b,8:F4}");
        }
    }

    // 4. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Matrices) ---");
    Console.WriteLine($"{"Learned parameters:",-20} a1 = {A[0, 0],9:F4} | a2 = {A[1, 0],9:F4} | a3 = {A[2, 0],9:F4} | b = {b,9:F4}");
    Console.WriteLine($"{"Expected parameters:",-20} a1 = {2,9:F4} | a2 = {3,9:F4} | a3 = {-1,9:F4} | b = {5,9:F4}");
}

void MatricesWithBias()
{
    // 1. Convert data to matrices with bias term

    // Number of samples and coefficients
    int n = data.Length;
    int numCoefficients = data[0].Length - 1; // Number of independent variables (3 in this case)

    float[,] XAnd1 = new float[n, numCoefficients + 1]; // +1 for bias term
    float[,] Y = new float[n, 1];

    // Prepare feature matrix XAnd1 with bias term and target vector Y
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < numCoefficients; j++)
        {
            XAnd1[i, j] = data[i][j];
        }
        XAnd1[i, numCoefficients] = 1; // Bias term
        Y[i, 0] = data[i][numCoefficients];
    }

    // 2. Initialize model parameters

    // These are the coefficients for our independent variables and the bias term
    float[,] AB = new float[numCoefficients + 1, 1]; // Corresponds to a1, a2, a3, b. It's already initialized to 0 at this point.

    // 3. Training loop

    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Prediction and error calculation

        // Make predictions for all samples at once: predictions = XAnd1 * AB
        float[,] predictions = XAnd1.MultiplyDot(AB);

        // Calculate errors for all samples: errors = Y - predictions
        float[,] errors = Y.Subtract(predictions);

        // Calculate gradient for coefficients 'AB': ∂MSE/∂AB = -2/n * XAnd1^T * errors
        // We can precalculate XAnd1.Transpose() and (-2.0f / n) for efficiency, but let's leave it as is for clarity.
        float[,] deltaAB = XAnd1.Transpose().MultiplyDot(errors).Multiply(-2.0f / n);

        // Update regression parameters using gradient descent
        AB = AB.Subtract(deltaAB.Multiply(LearningRate));

        if (iteration % PrintEvery == 0)
        {
            // Calculate the Mean Squared Error loss: MSE = mean(errors^2)
            float meanSquaredError = errors.Power(2).Mean();

            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5} | a1: {AB[0, 0],8:F4} | a2: {AB[1, 0],8:F4} | a3: {AB[2, 0],8:F4} | b: {AB[3, 0],8:F4}");
        }
    }

    // 4. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Matrices with Bias) ---");
    Console.WriteLine($"{"Learned parameters:",-20} a1 = {AB[0, 0],9:F4} | a2 = {AB[1, 0],9:F4} | a3 = {AB[2, 0],9:F4} | b = {AB[3, 0],9:F4}");
    Console.WriteLine($"{"Expected parameters:",-20} a1 = {2,9:F4} | a2 = {3,9:F4} | a3 = {-1,9:F4} | b = {5,9:F4}");
}

void MatricesWithBiasBoston()
{
    // 1. Load Boston housing dataset and standardize it (scale features to have mean 0 and stddev 1)

    float[,] bostonData = LoadCsv("..\\..\\..\\..\\..\\data\\Boston\\BostonHousing.csv");
    bostonData.Standardize();

    // 2. Convert data to matrices with bias term

    // Number of samples and coefficients
    int n = bostonData.GetLength(0);
    int numCoefficients = bostonData.GetLength(1) - 1; // Number of independent variables

    float[,] XAnd1 = new float[n, numCoefficients + 1]; // +1 for bias term
    float[,] Y = new float[n, 1];

    // Prepare feature matrix XAnd1 with bias term and target vector Y
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < numCoefficients; j++)
        {
            XAnd1[i, j] = bostonData[i, j];
        }
        XAnd1[i, numCoefficients] = 1; // Bias term
        Y[i, 0] = bostonData[i, numCoefficients];
    }

    // 2. Initialize model parameters

    // These are the coefficients for our independent variables and the bias term
    float[,] AB = new float[numCoefficients + 1, 1];

    // 3. Training loop

    for (int iteration = 1; iteration <= Iterations; iteration++)
    {
        // Prediction and error calculation

        // Make predictions for all samples at once: predictions = XAnd1 * AB
        float[,] predictions = XAnd1.MultiplyDot(AB);

        // Calculate errors for all samples: errors = Y - predictions
        float[,] errors = Y.Subtract(predictions);

        // Calculate gradient for coefficients 'AB': ∂MSE/∂AB = -2/n * XAnd1^T * errors
        // We can precalculate XAnd1.Transpose() and (-2.0f / n) for efficiency, but let's leave it as is for clarity.
        float[,] deltaAB = XAnd1.Transpose().MultiplyDot(errors).Multiply(-2.0f / n);

        // Update regression parameters using gradient descent
        AB = AB.Subtract(deltaAB.Multiply(LearningRate));

        if (iteration % PrintEvery == 0)
        {
            // Calculate the Mean Squared Error loss: MSE = mean(errors^2)
            float meanSquaredError = errors.Power(2).Mean();

            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5} | a1: {AB[0, 0],8:F4} | a2: {AB[1, 0],8:F4} | a3: {AB[2, 0],8:F4} | b: {AB[3, 0],8:F4}");
        }
    }

    // 4. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Matrices with Bias) ---");
    Console.WriteLine($"{"Learned parameters:",-20} a1 = {AB[0, 0],9:F4} | a2 = {AB[1, 0],9:F4} | a3 = {AB[2, 0],9:F4} | b = {AB[3, 0],9:F4}");
    Console.WriteLine($"{"Expected parameters:",-20} a1 = {2,9:F4} | a2 = {3,9:F4} | a3 = {-1,9:F4} | b = {5,9:F4}");
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