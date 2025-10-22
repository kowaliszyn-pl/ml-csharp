// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

// Set the parameters for the model

const float LearningRate = 0.0005f;
const int Iterations = 35_000;
const int PrintEvery = 1_000;

bool running = true;
Console.OutputEncoding = System.Text.Encoding.UTF8;

while (running)
{
    Console.WriteLine("Select a routine to run:");
    Console.WriteLine("1. Variables");
    Console.WriteLine("2. Tables");
    Console.WriteLine("3. Matrices");
    Console.WriteLine("4. Matrices with bias");
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
            // Matrices();
            break;
        case "4":
            // MatricesWithBias();
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
    // 1. Prepare training data
    // Each inner array represents a sample: [x1, x2, x3, y]
    // We are trying to find the relationship: y = 2*x1 + 3*x2 - 1*x3 + 5

    float[][] data = [
        [1, 2, 1, 12], // y = 2*1 + 3*2 - 1*1 + 5 = 12
        [2, 1, 2, 10], // etc.
        [3, 3, 1, 19], 
        [4, 2, 3, 16],
        [1, 4, 2, 17]  
    ];

    // 2. Initialize model parameters
    // These are the coefficients for our independent variables and the bias term

    float a1 = 0, a2 = 0, a3 = 0; // Parameters for x1, x2, x3
    float b = 0;

    // 3. Training loop

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

        // Number of samples
        int n = data.Length;

        // MSE
        float meanSquaredError = sumSquaredError / n;

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
            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5} | a1: {a1,8:F4} | a2: {a2,8:F4} | a3: {a3,8:F4} | b: {b,8:F4}");
        }
    }

    // 4. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Variables) ---");
    Console.WriteLine($"{"Learned parameters:",-20} a1 = {a1,9:F4} | a2 = {a2,9:F4} | a3 = {a3,9:F4} | b = {b,9:F4}");
    Console.WriteLine($"{"Expected parameters:",-20} a1 = {2,9:F4} | a2 = {3,9:F4} | a3 = {-1,9:F4} | b = {5,9:F4}");
}

void Tables()
{
    // 1. Prepare training data
    // Each inner array represents a sample: [x1, x2, x3, y]
    // We are trying to find the relationship: y = 2*x1 + 3*x2 - 1*x3 + 5

    float[][] data = [
        [1, 2, 1, 12], // y = 2*1 + 3*2 - 1*1 + 5 = 12
        [2, 1, 2, 10], // etc.
        [3, 3, 1, 19], 
        [4, 2, 3, 16], 
        [1, 4, 2, 17]  
    ];

    // Number of samples and coefficients
    int n = data.Length;
    int numCoefficients = data[0].Length - 1; // Number of independent variables (3 in this case)

    // 2. Initialize model parameters
    // These are the coefficients for our independent variables and the bias term

    float[] a = new float[numCoefficients]; // Corresponds to a1, a2, a3. It's already initialized to 0 at this point.
    float b = 0;

    // 3. Training loop

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

        // MSE
        float meanSquaredError = sumSquaredError / n;

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
            Console.WriteLine($"Iteration: {iteration,6} | MSE: {meanSquaredError,8:F5} | a1: {a[0],8:F4} | a2: {a[1],8:F4} | a3: {a[2],8:F4} | b: {b,8:F4}");
        }
    }

    // 4. Output learned parameters

    Console.WriteLine("\n--- Training Complete (Tables) ---");
    Console.WriteLine($"{"Learned parameters:",-20} a1 = {a[0],9:F4} | a2 = {a[1],9:F4} | a3 = {a[2],9:F4} | b = {b,9:F4}");
    Console.WriteLine($"{"Expected parameters:",-20} a1 = {2,9:F4} | a2 = {3,9:F4} | a3 = {-1,9:F4} | b = {5,9:F4}");
}