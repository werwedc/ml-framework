using NUnit.Framework;
using MLFramework.Core;
using MLFramework.Fusion;
using MLFramework.Fusion.Validation;
using MLFramework.Fusion.Backends;
using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;
using MLFramework.Autotuning;

namespace MLFramework.Tests.Fusion.Validation;

/// <summary>
/// Tests for FusionFallbackHandler
/// </summary>
[TestFixture]
public class FusionFallbackHandlerTests
{
    private MockLogger _logger;
    private MockKernelExecutor _executor;
    private FusionFallbackHandler _fallback;
    private TensorShape _defaultShape;

    [SetUp]
    public void Setup()
    {
        _logger = new MockLogger();
        _executor = new MockKernelExecutor();
        _fallback = new FusionFallbackHandler(_logger, _executor);
        _defaultShape = TensorShape.Create(1, 3, 224, 224);
    }

    [Test]
    public void ExecuteSeparate_SingleOperation_ExecutesKernel()
    {
        var ops = new[] { CreateValidOp() };
        var input = CreateTestTensor();

        var output = _fallback.ExecuteSeparate(ops, input);

        Assert.IsNotNull(output);
        Assert.AreEqual(1, _executor.ExecutionCount);
    }

    [Test]
    public void ExecuteSeparate_MultipleOperations_ExecutesAllKernels()
    {
        var ops = new[]
        {
            CreateValidOp(),
            CreateValidOp(),
            CreateValidOp()
        };
        var input = CreateTestTensor();

        var output = _fallback.ExecuteSeparate(ops, input);

        Assert.IsNotNull(output);
        Assert.AreEqual(3, _executor.ExecutionCount);
    }

    [Test]
    public void ExecuteSeparate_EmptyOperations_ThrowsException()
    {
        var ops = Array.Empty<Operation>();
        var input = CreateTestTensor();

        Assert.Throws<ArgumentException>(() => _fallback.ExecuteSeparate(ops, input));
    }

    [Test]
    public void LogFallbackReason_LogsWarning()
    {
        var ops = new[] { CreateValidOp(), CreateValidOp() };
        var reason = "Layout mismatch detected";

        _fallback.LogFallbackReason(reason, ops);

        Assert.IsTrue(_logger.Warnings.Any(w => w.Contains("Layout mismatch detected")));
    }

    [Test]
    public void LogFallbackReason_EmptyOperations_LogsWarning()
    {
        var ops = Array.Empty<Operation>();
        var reason = "Test reason";

        _fallback.LogFallbackReason(reason, ops);

        Assert.IsTrue(_logger.Warnings.Any(w => w.Contains("Test reason")));
    }

    [Test]
    public void ExecuteWithFallback_HasViolations_LogsAndExecutes()
    {
        var ops = new[] { CreateValidOp(), CreateValidOp() };
        var input = CreateTestTensor();
        var violations = new[]
        {
            new ConstraintViolation
            {
                ConstraintName = "Test",
                Message = "Test violation",
                Severity = Severity.Error
            }
        };

        var output = _fallback.ExecuteWithFallback(ops, input, violations);

        Assert.IsNotNull(output);
        Assert.IsTrue(_logger.Warnings.Any(w => w.Contains("Test violation")));
        Assert.AreEqual(2, _executor.ExecutionCount);
    }

    [Test]
    public void ExecuteWithFallback_NoViolations_ExecutesWithoutLogging()
    {
        var ops = new[] { CreateValidOp(), CreateValidOp() };
        var input = CreateTestTensor();
        var violations = Array.Empty<ConstraintViolation>();

        var output = _fallback.ExecuteWithFallback(ops, input, violations);

        Assert.IsNotNull(output);
        Assert.IsFalse(_logger.Warnings.Any());
    }

    #region Helper Methods

    private Operation CreateValidOp()
    {
        return new TestOperation
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

    #endregion

    #region Mock Classes

    private class MockLogger : ILogger
    {
        public List<string> Warnings { get; } = new();

        public void LogInformation(string message, params object[] args)
        {
        }

        public void LogWarning(string message, params object[] args)
        {
            Warnings.Add(string.Format(message, args));
        }

        public void LogError(string message, params object[] args)
        {
        }
    }

    private class MockKernelExecutor : IKernelExecutor
    {
        public int ExecutionCount { get; private set; } = 0;

        public void ExecuteFusedKernel(
            MLFramework.Fusion.FusedOperation fusedOp,
            KernelLaunchConfiguration config,
            Tensor input)
        {
            ExecutionCount++;
        }

        public Tensor ExecuteKernel(Operation op, Tensor input)
        {
            ExecutionCount++;
            return input; // Return input as output for testing
        }

        public void Synchronize()
        {
        }
    }

    #endregion
}
