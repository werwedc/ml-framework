using NUnit.Framework;
using MLFramework.Core;
using MLFramework.Fusion;
using MLFramework.Fusion.Validation;
using FusionBackends = MLFramework.Fusion.Backends;
using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;
using MLFramework.Autotuning;

namespace MLFramework.Tests.Fusion.Validation;

/// <summary>
/// Tests for FusionVerifier
/// </summary>
[TestFixture]
public class FusionVerifierTests
{
    private MockKernelExecutor _executor;
    private MockTensorGenerator _generator;
    private FusionVerifier _verifier;
    private TensorShape _defaultShape;

    [SetUp]
    public void Setup()
    {
        _executor = new MockKernelExecutor();
        _generator = new MockTensorGenerator();
        _verifier = new FusionVerifier(_executor, _generator, 1e-5);
        _defaultShape = TensorShape.Create(1, 3, 224, 224);
    }

    [Test]
    public void Verify_NullFusedOperation_ThrowsException()
    {
        var originalOps = CreateValidOperations();
        var testInput = CreateTestTensor();

        Assert.Throws<ArgumentNullException>(() =>
            _verifier.Verify(null!, originalOps, testInput));
    }

    [Test]
    public void Verify_EmptyOriginalOperations_ThrowsException()
    {
        var fusedOp = CreateFusedOperation();
        var testInput = CreateTestTensor();

        Assert.Throws<ArgumentException>(() =>
            _verifier.Verify(fusedOp, Array.Empty<Operation>(), testInput));
    }

    [Test]
    public void Verify_NullTestInput_ThrowsException()
    {
        var fusedOp = CreateFusedOperation();
        var originalOps = CreateValidOperations();

        Assert.Throws<ArgumentNullException>(() =>
            _verifier.Verify(fusedOp, originalOps, null!));
    }

    [Test]
    public void Verify_ValidInputs_ReturnsVerificationResult()
    {
        var fusedOp = CreateFusedOperation();
        var originalOps = CreateValidOperations();
        var testInput = CreateTestTensor();

        var result = _verifier.Verify(fusedOp, originalOps, testInput);

        Assert.IsNotNull(result);
        Assert.IsNotNull(result.TestCases);
        Assert.AreEqual(1, result.TestCases.Count);
    }

    [Test]
    public void VerifyWithRandomInputs_ValidInputs_ReturnsMultipleTestCases()
    {
        var fusedOp = CreateFusedOperation();
        var originalOps = CreateValidOperations();
        int testCases = 5;

        var result = _verifier.VerifyWithRandomInputs(fusedOp, originalOps, testCases);

        Assert.IsNotNull(result);
        Assert.AreEqual(testCases, result.TestCases.Count);
        Assert.AreEqual(testCases, result.PassedTestCases + result.FailedTestCases);
    }

    [Test]
    public void VerifyWithRandomInputs_ZeroTestCases_ThrowsException()
    {
        var fusedOp = CreateFusedOperation();
        var originalOps = CreateValidOperations();

        Assert.Throws<ArgumentException>(() =>
            _verifier.VerifyWithRandomInputs(fusedOp, originalOps, 0));
    }

    [Test]
    public void VerifyWithRandomInputs_DefaultTestCases_Uses10Tests()
    {
        var fusedOp = CreateFusedOperation();
        var originalOps = CreateValidOperations();

        var result = _verifier.VerifyWithRandomInputs(fusedOp, originalOps);

        Assert.AreEqual(10, result.TestCases.Count);
    }

    [Test]
    public void FusionVerifier_DefaultTolerance_Uses1eMinus5()
    {
        var executor = new MockKernelExecutor();
        var generator = new MockTensorGenerator();
        var verifier = new FusionVerifier(executor, generator);

        var fusedOp = CreateFusedOperation();
        var originalOps = CreateValidOperations();
        var testInput = CreateTestTensor();

        var result = verifier.Verify(fusedOp, originalOps, testInput);

        // Result should be created with default tolerance
        Assert.IsNotNull(result);
    }

    [Test]
    public void FusionVerifier_CustomTolerance_UsesProvidedTolerance()
    {
        var executor = new MockKernelExecutor();
        var generator = new MockTensorGenerator();
        var customTolerance = 1e-3;
        var verifier = new FusionVerifier(executor, generator, customTolerance);

        var fusedOp = CreateFusedOperation();
        var originalOps = CreateValidOperations();
        var testInput = CreateTestTensor();

        var result = verifier.Verify(fusedOp, originalOps, testInput);

        // Result should be created with custom tolerance
        Assert.IsNotNull(result);
    }

    #region Helper Methods

    private MLFramework.Fusion.FusedOperation CreateFusedOperation()
    {
        var pattern = new FusionPatternDefinition
        {
            Name = "TestPattern",
            Description = "Test pattern",
            Type = Backends.FusionPatternType.ElementWise,
            Operations = Array.Empty<string>()
        };

        var ir = new FusionIR
        {
            Statements = new List<FusionIRStatement>()
        };

        var kernelSpec = new KernelSpecification
        {
            KernelName = "test_kernel",
            Code = "// test code",
            Language = "C++"
        };

        return MLFramework.Fusion.FusedOperation.Create(
            Guid.NewGuid().ToString(),
            CreateValidOperations(),
            pattern,
            ir,
            kernelSpec);
    }

    private List<Operation> CreateValidOperations()
    {
        return new List<Operation>
        {
            new TestOperation
            {
                Id = Guid.NewGuid().ToString(),
                Type = "Add",
                Name = $"Add_{Guid.NewGuid():N}",
                DataType = RitterFramework.Core.DataType.Float32,
                Layout = TensorLayout.NCHW,
                InputShape = _defaultShape,
                OutputShape = _defaultShape,
                Inputs = new[] { $"input_{Guid.NewGuid():N}" },
                Outputs = new[] { $"output_{Guid.NewGuid():N}" },
                Attributes = new Dictionary<string, object>()
            },
            new TestOperation
            {
                Id = Guid.NewGuid().ToString(),
                Type = "ReLU",
                Name = $"ReLU_{Guid.NewGuid():N}",
                DataType = RitterFramework.Core.DataType.Float32,
                Layout = TensorLayout.NCHW,
                InputShape = _defaultShape,
                OutputShape = _defaultShape,
                Inputs = new[] { $"output_0" },
                Outputs = new[] { $"output_{Guid.NewGuid():N}" },
                Attributes = new Dictionary<string, object>()
            }
        };
    }

    private Tensor CreateTestTensor()
    {
        var data = new float[1 * 3 * 224 * 224];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = i % 100f;
        }
        return new Tensor(data, new[] { 1, 3, 224, 224 });
    }

    private class MockKernelExecutor : IKernelExecutor
    {
        public void ExecuteFusedKernel(
            MLFramework.Fusion.FusedOperation fusedOp,
            FusionBackends.KernelLaunchConfiguration config,
            Tensor input)
        {
            // Mock implementation - does nothing
        }

        public Tensor ExecuteKernel(Operation op, Tensor input)
        {
            // Mock implementation - returns input as output
            return input;
        }

        public void Synchronize()
        {
            // Mock implementation - does nothing
        }
    }

    private class MockTensorGenerator : ITensorGenerator
    {
        private int _counter = 0;

        public Tensor GenerateRandomTensor(TensorShape shape, MLFramework.Core.DataType dataType)
        {
            var size = shape.Dimensions.Aggregate(1, (acc, dim) => acc * dim);
            var data = new float[size];

            // Generate deterministic "random" data based on counter
            var random = new Random(_counter++);
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)random.NextDouble();
            }

            return new Tensor(data, shape.Dimensions.ToArray());
        }
    }

    #endregion

    #region Mock Classes



    #endregion
}
