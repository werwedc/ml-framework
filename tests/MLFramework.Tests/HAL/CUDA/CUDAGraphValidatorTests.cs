using NUnit.Framework;
using MLFramework.HAL.CUDA;
using MLFramework.HAL.CUDA.Graphs.Validation;
using MLFramework.HAL.CUDA.Graphs.Validation.Rules;

namespace MLFramework.Tests.HAL.CUDA;

/// <summary>
/// Tests for CUDA Graph Validation System
/// </summary>
[TestFixture]
public class CUDAGraphValidatorTests
{
    private CUDAGraphValidator? _validator;

    [SetUp]
    public void Setup()
    {
        _validator = new CUDAGraphValidator();
    }

    [TearDown]
    public void TearDown()
    {
        // Validator doesn't need disposal
    }

    [Test]
    public void Constructor_InitializesWithDefaultRules()
    {
        Assert.IsNotNull(_validator);
        var rules = _validator!.GetRules();
        Assert.IsNotNull(rules);
        Assert.IsNotEmpty(rules);
        Assert.AreEqual(5, rules.Count); // 5 default rules
    }

    [Test]
    public void Validate_NullGraph_ReturnsError()
    {
        var result = _validator!.Validate(null!);

        Assert.IsFalse(result.IsValid);
        Assert.IsNotEmpty(result.Errors);
        Assert.That(result.Errors[0], Does.Contain("null"));
    }

    [Test]
    public void Validate_WithCustomRule_AddsRuleToValidator()
    {
        var customRule = new CustomTestRule();
        _validator!.RegisterRule(customRule);

        var rules = _validator.GetRules();
        Assert.AreEqual(6, rules.Count); // 5 default + 1 custom
    }

    [Test]
    public void RegisterRule_NullRule_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => _validator!.RegisterRule(null!));
    }

    [Test]
    public void ClearRules_RemovesAllRules()
    {
        _validator!.ClearRules();

        var rules = _validator.GetRules();
        Assert.IsEmpty(rules);
    }

    [Test]
    public void Validate_WithEmptyGraphRule_ReturnsErrorForEmptyGraph()
    {
        var mockGraph = new MockEmptyGraph();
        var result = _validator!.Validate(mockGraph);

        Assert.IsFalse(result.IsValid);
        Assert.That(result.Errors, Has.Some.Contains("no operations"));
    }

    [Test]
    public void Validate_WithWarnings_ReturnsValidWithWarnings()
    {
        var mockGraph = new MockValidGraph();
        var result = _validator!.Validate(mockGraph);

        // Should have warnings from built-in rules but still be valid
        Assert.IsTrue(result.IsValid);
        Assert.IsNotEmpty(result.Warnings);
    }

    [Test]
    public void ValidateGraphInstance_WithNullHandle_ReturnsError()
    {
        var result = _validator!.ValidateGraphInstance(IntPtr.Zero);

        Assert.IsFalse(result.IsValid);
        Assert.IsNotEmpty(result.Errors);
        Assert.That(result.Errors[0], Does.Contain("null"));
    }
}

/// <summary>
/// Tests for CUDAGraphValidationContext
/// </summary>
[TestFixture]
public class CUDAGraphValidationContextTests
{
    [Test]
    public void Constructor_InitializesEmptyCollections()
    {
        var context = new CUDAGraphValidationContext();

        Assert.IsNotNull(context.CapturedOperations);
        Assert.IsEmpty(context.CapturedOperations);
        Assert.IsNotNull(context.AllocatedMemory);
        Assert.IsEmpty(context.AllocatedMemory);
    }

    [Test]
    public void RecordOperation_AddsOperationToList()
    {
        var context = new CUDAGraphValidationContext();

        context.RecordOperation("kernel_launch");

        Assert.AreEqual(1, context.CapturedOperations.Count);
        Assert.AreEqual("kernel_launch", context.CapturedOperations[0]);
    }

    [Test]
    public void RecordOperation_NullOrEmpty_DoesNotAdd()
    {
        var context = new CUDAGraphValidationContext();

        context.RecordOperation(null!);
        context.RecordOperation("");
        context.RecordOperation("   ");

        Assert.IsEmpty(context.CapturedOperations);
    }

    [Test]
    public void RecordOperation_MultipleOperations_TracksCounts()
    {
        var context = new CUDAGraphValidationContext();

        context.RecordOperation("kernel_launch");
        context.RecordOperation("memcpy");
        context.RecordOperation("kernel_launch");

        Assert.AreEqual(3, context.CapturedOperations.Count);
        Assert.AreEqual(2, context.GetOperationCount("kernel_launch"));
        Assert.AreEqual(1, context.GetOperationCount("memcpy"));
    }

    [Test]
    public void RecordMemoryAllocation_AddsPointer()
    {
        var context = new CUDAGraphValidationContext();
        var ptr = new IntPtr(0x12345678);

        context.RecordMemoryAllocation(ptr);

        Assert.IsTrue(context.AllocatedMemory.Contains(ptr));
    }

    [Test]
    public void RecordMemoryAllocation_ZeroPointer_DoesNotAdd()
    {
        var context = new CUDAGraphValidationContext();

        context.RecordMemoryAllocation(IntPtr.Zero);

        Assert.IsFalse(context.AllocatedMemory.Contains(IntPtr.Zero));
    }

    [Test]
    public void HasOperation_ReturnsTrueForExistingOperation()
    {
        var context = new CUDAGraphValidationContext();

        context.RecordOperation("kernel_launch");

        Assert.IsTrue(context.HasOperation("kernel_launch"));
        Assert.IsFalse(context.HasOperation("memcpy"));
    }

    [Test]
    public void Clear_ResetsAllCollections()
    {
        var context = new CUDAGraphValidationContext();

        context.RecordOperation("kernel_launch");
        context.RecordMemoryAllocation(new IntPtr(0x12345678));
        context.Clear();

        Assert.IsEmpty(context.CapturedOperations);
        Assert.IsEmpty(context.AllocatedMemory);
    }
}

/// <summary>
/// Tests for individual validation rules
/// </summary>
[TestFixture]
public class ValidationRuleTests
{
    [Test]
    public void EmptyGraphRule_WithEmptyGraph_ReturnsError()
    {
        var rule = new EmptyGraphRule();
        var mockGraph = new MockEmptyGraph();

        var result = rule.Validate(mockGraph);

        Assert.IsFalse(result.IsValid);
        Assert.IsNotEmpty(result.Errors);
        Assert.That(result.Errors[0], Does.Contain("no operations"));
    }

    [Test]
    public void EmptyGraphRule_WithNullGraph_ReturnsError()
    {
        var rule = new EmptyGraphRule();

        var result = rule.Validate(null!);

        Assert.IsFalse(result.IsValid);
        Assert.That(result.Errors[0], Does.Contain("null"));
    }

    [Test]
    public void DynamicMemoryRule_AlwaysReturnsWarning()
    {
        var rule = new DynamicMemoryRule();
        var mockGraph = new MockValidGraph();

        var result = rule.Validate(mockGraph);

        Assert.IsTrue(result.IsValid); // Warnings don't make it invalid
        Assert.IsNotEmpty(result.Warnings);
        Assert.That(result.Warnings[0], Does.Contain("memory pool"));
    }

    [Test]
    public void ControlFlowRule_AlwaysReturnsWarning()
    {
        var rule = new ControlFlowRule();
        var mockGraph = new MockValidGraph();

        var result = rule.Validate(mockGraph);

        Assert.IsTrue(result.IsValid);
        Assert.IsNotEmpty(result.Warnings);
        Assert.That(result.Warnings[0], Does.Contain("control flow"));
    }

    [Test]
    public void SynchronizationRule_AlwaysReturnsWarning()
    {
        var rule = new SynchronizationRule();
        var mockGraph = new MockValidGraph();

        var result = rule.Validate(mockGraph);

        Assert.IsTrue(result.IsValid);
        Assert.IsNotEmpty(result.Warnings);
        Assert.That(result.Warnings[0], Does.Contain("synchronization"));
    }

    [Test]
    public void IOOperationRule_AlwaysReturnsWarning()
    {
        var rule = new IOOperationRule();
        var mockGraph = new MockValidGraph();

        var result = rule.Validate(mockGraph);

        Assert.IsTrue(result.IsValid);
        Assert.IsNotEmpty(result.Warnings);
        Assert.That(result.Warnings[0], Does.Contain("I/O"));
    }
}

#region Mock Implementations

/// <summary>
/// Mock ICUDAGraph for testing - empty graph
/// </summary>
internal class MockEmptyGraph : ICUDAGraph
{
    public string GraphId => "mock-empty-graph";
    public CUDAGraphState State => CUDAGraphState.Created;

    public void Execute(CudaStream stream)
    {
        throw new NotImplementedException();
    }

    public CUDAGraphValidationResult Validate()
    {
        return CUDAGraphValidationResult.Success(0);
    }

    public void Dispose()
    {
        // No-op for mock
    }
}

/// <summary>
/// Mock ICUDAGraph for testing - valid graph with operations
/// </summary>
internal class MockValidGraph : ICUDAGraph
{
    public string GraphId => "mock-valid-graph";
    public CUDAGraphState State => CUDAGraphState.Ready;

    public void Execute(CudaStream stream)
    {
        // No-op for mock
    }

    public CUDAGraphValidationResult Validate()
    {
        return CUDAGraphValidationResult.Success(5);
    }

    public void Dispose()
    {
        // No-op for mock
    }
}

/// <summary>
/// Custom validation rule for testing rule registration
/// </summary>
internal class CustomTestRule : IValidationRule
{
    public string RuleName => "CustomTestRule";
    public string Description => "A custom test rule";

    public ValidationResult Validate(ICUDAGraph graph)
    {
        return new ValidationResult();
    }
}

#endregion
