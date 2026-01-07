using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;
using MLFramework.Autotuning;
using MLFramework.Fusion.Backends;

namespace MLFramework.Fusion.Validation;

/// <summary>
/// Verifies that fused operations produce correct results
/// </summary>
public class FusionVerifier : IFusionVerifier
{
    private readonly IKernelExecutor _executor;
    private readonly ITensorGenerator _generator;
    private readonly double _tolerance;

    /// <summary>
    /// Creates a new FusionVerifier with default tolerance (1e-5)
    /// </summary>
    public FusionVerifier(IKernelExecutor executor, ITensorGenerator generator)
        : this(executor, generator, 1e-5)
    {
    }

    /// <summary>
    /// Creates a new FusionVerifier with custom tolerance
    /// </summary>
    public FusionVerifier(IKernelExecutor executor, ITensorGenerator generator, double tolerance)
    {
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _tolerance = tolerance;
    }

    /// <inheritdoc/>
    public VerificationResult Verify(
        FusedOperation fusedOp,
        IReadOnlyList<Operation> originalOps,
        Tensor testInput)
    {
        if (fusedOp == null)
            throw new ArgumentNullException(nameof(fusedOp));
        if (originalOps == null || originalOps.Count == 0)
            throw new ArgumentException("Original operations cannot be null or empty", nameof(originalOps));
        if (testInput == null)
            throw new ArgumentNullException(nameof(testInput));

        // Execute fused operation
        var config = new KernelLaunchConfiguration
        {
            BlockDim = new Backends.ThreadBlockConfiguration { X = 256, Y = 1, Z = 1 },
            GridDim = new Backends.ThreadBlockConfiguration { X = 1, Y = 1, Z = 1 },
            SharedMemoryBytes = 0,
            Parameters = Array.Empty<Backends.KernelLaunchParameter>()
        };

        _executor.ExecuteFusedKernel(fusedOp, config, testInput);
        _executor.Synchronize();

        // Note: In a real implementation, the ExecuteFusedKernel would return the output tensor.
        // For now, we'll create a placeholder for demonstration.
        var fusedOutput = testInput; // Placeholder - should be actual output

        // Execute sequential operations
        var sequentialOutput = testInput;
        foreach (var op in originalOps)
        {
            sequentialOutput = _executor.ExecuteKernel(op, sequentialOutput);
        }
        _executor.Synchronize();

        // Compare results
        var error = ComputeError(fusedOutput, sequentialOutput);
        var tolerancePassed = error < _tolerance;

        return new VerificationResult
        {
            Passed = tolerancePassed,
            MaxError = error,
            MeanError = error,
            TestCases = new[]
            {
                new VerificationTestResult
                {
                    TestCaseNumber = 1,
                    FusedOutput = fusedOutput,
                    SequentialOutput = sequentialOutput,
                    Error = error,
                    TolerancePassed = tolerancePassed
                }
            }
        };
    }

    /// <inheritdoc/>
    public VerificationResult VerifyWithRandomInputs(
        FusedOperation fusedOp,
        IReadOnlyList<Operation> originalOps,
        int testCases = 10)
    {
        if (fusedOp == null)
            throw new ArgumentNullException(nameof(fusedOp));
        if (originalOps == null || originalOps.Count == 0)
            throw new ArgumentException("Original operations cannot be null or empty", nameof(originalOps));
        if (testCases <= 0)
            throw new ArgumentException("Test cases must be positive", nameof(testCases));

        var testResults = new List<VerificationTestResult>();
        var maxError = 0.0;
        var totalError = 0.0;

        for (int i = 0; i < testCases; i++)
        {
            var testInput = _generator.GenerateRandomTensor(
                fusedOp.InputShape,
                fusedOp.DataType);

            var result = Verify(fusedOp, originalOps, testInput);
            var testCase = result.TestCases[0];

            testResults.Add(testCase with { TestCaseNumber = i + 1 });
            maxError = Math.Max(maxError, testCase.Error);
            totalError += testCase.Error;
        }

        return new VerificationResult
        {
            Passed = maxError < _tolerance,
            MaxError = maxError,
            MeanError = totalError / testCases,
            TestCases = testResults
        };
    }

    /// <summary>
    /// Computes the maximum absolute error between two tensors
    /// </summary>
    private double ComputeError(Tensor a, Tensor b)
    {
        if (a.Shape.Length != b.Shape.Length)
            throw new ArgumentException("Tensor shapes must match for error computation");

        double maxError = 0;
        var elements = a.Shape.Aggregate(1, (acc, dim) => acc * dim);

        for (int i = 0; i < elements; i++)
        {
            var diff = Math.Abs(a.Data[i] - b.Data[i]);
            maxError = Math.Max(maxError, diff);
        }

        return maxError;
    }
}
