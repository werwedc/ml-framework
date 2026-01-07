using NUnit.Framework;
using MLFramework.Core;
using MLFramework.Fusion;
using MLFramework.Fusion.Validation;

namespace MLFramework.Tests.Fusion.Validation;

/// <summary>
/// Tests for FusionConstraintsValidator
/// </summary>
[TestFixture]
public class FusionConstraintsValidatorTests
{
    private TensorShape _defaultShape;

    [SetUp]
    public void Setup()
    {
        _defaultShape = TensorShape.Create(1, 3, 224, 224);
    }

    [Test]
    public void Validate_AllValid_ReturnsTrue()
    {
        var ops = new[]
        {
            CreateValidOp(),
            CreateValidOp()
        };

        var validator = new FusionConstraintsValidator();
        var result = validator.Validate(ops, out var violations);

        Assert.IsTrue(result);
        Assert.AreEqual(0, violations.Count);
    }

    [Test]
    public void Validate_LayoutMismatch_ReturnsFalse()
    {
        var ops = new[]
        {
            CreateValidOp(),
            CreateOpWithLayout(TensorLayout.NHWC)
        };

        var validator = new FusionConstraintsValidator();
        var result = validator.Validate(ops, out var violations);

        Assert.IsFalse(result);
        Assert.IsTrue(violations.Any(v => v.ConstraintName == "MemoryLayout"));
    }

    [Test]
    public void Validate_MultipleViolations_ReturnsFalse()
    {
        var ops = new[]
        {
            CreateOpWithLayout(TensorLayout.NHWC),
            CreateOpWithSideEffect("Print")
        };

        var validator = new FusionConstraintsValidator();
        var result = validator.Validate(ops, out var violations);

        Assert.IsFalse(result);
        Assert.Greater(violations.Count, 0);
    }

    [Test]
    public void CanFuse_AllValid_ReturnsTrue()
    {
        var ops = new[]
        {
            CreateValidOp(),
            CreateValidOp()
        };

        var validator = new FusionConstraintsValidator();
        var result = validator.CanFuse(ops);

        Assert.IsTrue(result);
    }

    [Test]
    public void CanFuse_HasErrors_ReturnsFalse()
    {
        var ops = new[]
        {
            CreateOpWithLayout(TensorLayout.NHWC),
            CreateValidOp()
        };

        var validator = new FusionConstraintsValidator();
        var result = validator.CanFuse(ops);

        Assert.IsFalse(result);
    }

    [Test]
    public void GetErrorViolations_MixedSeverities_ReturnsOnlyErrors()
    {
        var ops = new[]
        {
            CreateValidOp(),
            CreateOpWithType("Gather") // Warning for gather
        };

        var validator = new FusionConstraintsValidator();
        var violations = validator.GetErrorViolations(ops);

        Assert.IsFalse(violations.Any(v => v.Severity == Severity.Warning));
    }

    [Test]
    public void GetWarningViolations_MixedSeverities_ReturnsOnlyWarnings()
    {
        var ops = new[]
        {
            CreateValidOp(),
            CreateOpWithType("Gather") // Warning for gather
        };

        var validator = new FusionConstraintsValidator();
        var violations = validator.GetWarningViolations(ops);

        Assert.IsTrue(violations.All(v => v.Severity == Severity.Warning));
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

    private Operation CreateOpWithLayout(TensorLayout layout)
    {
        return new TestOperation
        {
            Id = Guid.NewGuid().ToString(),
            Type = "TestOp",
            Name = $"TestOp_{Guid.NewGuid():N}",
            DataType = RitterFramework.Core.DataType.Float32,
            Layout = layout,
            InputShape = _defaultShape,
            OutputShape = _defaultShape,
            Inputs = new[] { $"input_{Guid.NewGuid():N}" },
            Outputs = new[] { $"output_{Guid.NewGuid():N}" },
            Attributes = new Dictionary<string, object>()
        };
    }

    private Operation CreateOpWithSideEffect(string type)
    {
        return new TestOperation
        {
            Id = Guid.NewGuid().ToString(),
            Type = type,
            Name = $"{type}_{Guid.NewGuid():N}",
            DataType = RitterFramework.Core.DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = _defaultShape,
            OutputShape = _defaultShape,
            Inputs = new[] { $"input_{Guid.NewGuid():N}" },
            Outputs = new[] { $"output_{Guid.NewGuid():N}" },
            Attributes = new Dictionary<string, object>()
        };
    }

    private Operation CreateOpWithType(string type)
    {
        return new TestOperation
        {
            Id = Guid.NewGuid().ToString(),
            Type = type,
            Name = $"{type}_{Guid.NewGuid():N}",
            DataType = RitterFramework.Core.DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = _defaultShape,
            OutputShape = _defaultShape,
            Inputs = new[] { $"input_{Guid.NewGuid():N}" },
            Outputs = new[] { $"output_{Guid.NewGuid():N}" },
            Attributes = new Dictionary<string, object>()
        };
    }

    #endregion
}
