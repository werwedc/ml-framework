using NUnit.Framework;
using MLFramework.Core;
using MLFramework.Fusion;
using MLFramework.Fusion.Validation;

namespace MLFramework.Tests.Fusion.Validation;

/// <summary>
/// Tests for constraint validators
/// </summary>
[TestFixture]
public class ConstraintValidatorTests
{
    private TensorShape _defaultShape;

    [SetUp]
    public void Setup()
    {
        _defaultShape = TensorShape.Create(1, 3, 224, 224);
    }

    [Test]
    public void MemoryLayout_ValidLayouts_NoViolations()
    {
        var ops = new[]
        {
            CreateOpWithLayout(TensorLayout.NCHW),
            CreateOpWithLayout(TensorLayout.NCHW)
        };

        var validator = new MemoryLayoutConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(0, violations.Count);
    }

    [Test]
    public void MemoryLayout_LayoutMismatch_ReturnsError()
    {
        var ops = new[]
        {
            CreateOpWithLayout(TensorLayout.NCHW),
            CreateOpWithLayout(TensorLayout.NHWC)
        };

        var validator = new MemoryLayoutConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(1, violations.Count);
        Assert.AreEqual("MemoryLayout", violations[0].ConstraintName);
        Assert.AreEqual(Severity.Error, violations[0].Severity);
    }

    [Test]
    public void MemoryLayout_AnyLayout_NoViolations()
    {
        var ops = new[]
        {
            CreateOpWithLayout(TensorLayout.Any),
            CreateOpWithLayout(TensorLayout.NCHW)
        };

        var validator = new MemoryLayoutConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(0, violations.Count);
    }

    [Test]
    public void NumericalPrecision_SameDtype_NoViolations()
    {
        var ops = new[]
        {
            CreateOpWithDtype(RitterFramework.Core.DataType.Float32),
            CreateOpWithDtype(RitterFramework.Core.DataType.Float32)
        };

        var validator = new NumericalPrecisionConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(0, violations.Count);
    }

    [Test]
    public void NumericalPrecision_DtypeMismatch_ReturnsError()
    {
        var ops = new[]
        {
            CreateOpWithDtype(RitterFramework.Core.DataType.Float32),
            CreateOpWithDtype(RitterFramework.Core.DataType.Float16)
        };

        var validator = new NumericalPrecisionConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(1, violations.Count);
        Assert.AreEqual("NumericalPrecision", violations[0].ConstraintName);
        Assert.AreEqual(Severity.Error, violations[0].Severity);
    }

    [Test]
    public void NumericalPrecision_FP16WithReduce_ReturnsWarning()
    {
        var ops = new[]
        {
            CreateOpWithTypeAndDtype("ReduceSum", RitterFramework.Core.DataType.Float16),
            CreateOpWithDtype(RitterFramework.Core.DataType.Float16)
        };

        var validator = new NumericalPrecisionConstraint();
        var violations = validator.GetViolations(ops);

        var warning = violations.FirstOrDefault(v => v.Severity == Severity.Warning);
        Assert.IsNotNull(warning);
        Assert.AreEqual("NumericalPrecision", warning.ConstraintName);
    }

    [Test]
    public void ThreadBlock_ValidThreads_NoViolations()
    {
        var shape = TensorShape.Create(1, 3, 32, 32);
        var ops = new[]
        {
            CreateOpWithShape(shape),
            CreateOpWithShape(shape)
        };

        var validator = new ThreadBlockConstraint(maxThreadsPerBlock: 1024);
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(0, violations.Count);
    }

    [Test]
    public void ThreadBlock_ExceedsMaxThreads_ReturnsError()
    {
        var shape = TensorShape.Create(1, 3, 512, 512);
        var ops = new[]
        {
            CreateOpWithShape(shape)
        };

        var validator = new ThreadBlockConstraint(maxThreadsPerBlock: 1000);
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(1, violations.Count);
        Assert.AreEqual("ThreadBlock", violations[0].ConstraintName);
        Assert.AreEqual(Severity.Error, violations[0].Severity);
    }

    [Test]
    public void SideEffect_NoSideEffects_NoViolations()
    {
        var ops = new[]
        {
            CreateOpWithType("Add"),
            CreateOpWithType("Mul")
        };

        var validator = new SideEffectConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(0, violations.Count);
    }

    [Test]
    public void SideEffect_HasSideEffects_ReturnsError()
    {
        var ops = new[]
        {
            CreateOpWithType("Print"),
            CreateOpWithType("Add")
        };

        var validator = new SideEffectConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(1, violations.Count);
        Assert.AreEqual("SideEffect", violations[0].ConstraintName);
        Assert.IsTrue(violations[0].Message.Contains("Print"));
    }

    [Test]
    public void ControlFlow_NoComplexFlow_NoViolations()
    {
        var ops = new[]
        {
            CreateOpWithType("Add"),
            CreateOpWithType("ReLU")
        };

        var validator = new ControlFlowConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(0, violations.Count);
    }

    [Test]
    public void ControlFlow_DataDependentFlow_ReturnsError()
    {
        var ops = new[]
        {
            CreateOpWithType("Where"),
            CreateOpWithType("Add")
        };

        var validator = new ControlFlowConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(1, violations.Count);
        Assert.AreEqual("ControlFlow", violations[0].ConstraintName);
        Assert.IsTrue(violations[0].Message.Contains("data-dependent"));
    }

    [Test]
    public void ControlFlow_ComplexBranching_ReturnsError()
    {
        var ops = new[]
        {
            CreateOpWithType("Loop"),
            CreateOpWithType("Add")
        };

        var validator = new ControlFlowConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(1, violations.Count);
        Assert.AreEqual("ControlFlow", violations[0].ConstraintName);
        Assert.IsTrue(violations[0].Message.Contains("branching"));
    }

    [Test]
    public void MemoryAccessPattern_CompatiblePatterns_NoViolations()
    {
        var ops = new[]
        {
            CreateOpWithType("Add"),
            CreateOpWithType("ReLU")
        };

        var validator = new MemoryAccessPatternConstraint();
        var violations = validator.GetViolations(ops);

        Assert.AreEqual(0, violations.Count);
    }

    [Test]
    public void MemoryAccessPattern_Gather_ReturnsWarning()
    {
        var ops = new[]
        {
            CreateOpWithType("Gather"),
            CreateOpWithType("Add")
        };

        var validator = new MemoryAccessPatternConstraint();
        var violations = validator.GetViolations(ops);

        var warning = violations.FirstOrDefault(v => v.Severity == Severity.Warning);
        Assert.IsNotNull(warning);
        Assert.AreEqual("MemoryAccessPattern", warning.ConstraintName);
        Assert.IsTrue(warning.Message.Contains("Gather"));
    }

    [Test]
    public void MemoryAccessPattern_Scatter_ReturnsWarning()
    {
        var ops = new[]
        {
            CreateOpWithType("Scatter"),
            CreateOpWithType("Add")
        };

        var validator = new MemoryAccessPatternConstraint();
        var violations = validator.GetViolations(ops);

        var warning = violations.FirstOrDefault(v => v.Severity == Severity.Warning);
        Assert.IsNotNull(warning);
        Assert.AreEqual("MemoryAccessPattern", warning.ConstraintName);
        Assert.IsTrue(warning.Message.Contains("Scatter"));
    }

    #region Helper Methods

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

    private Operation CreateOpWithDtype(RitterFramework.Core.DataType dtype)
    {
        return new TestOperation
        {
            Id = Guid.NewGuid().ToString(),
            Type = "TestOp",
            Name = $"TestOp_{Guid.NewGuid():N}",
            DataType = dtype,
            Layout = TensorLayout.NCHW,
            InputShape = _defaultShape,
            OutputShape = _defaultShape,
            Inputs = new[] { $"input_{Guid.NewGuid():N}" },
            Outputs = new[] { $"output_{Guid.NewGuid():N}" },
            Attributes = new Dictionary<string, object>()
        };
    }

    private Operation CreateOpWithShape(TensorShape shape)
    {
        return new TestOperation
        {
            Id = Guid.NewGuid().ToString(),
            Type = "TestOp",
            Name = $"TestOp_{Guid.NewGuid():N}",
            DataType = RitterFramework.Core.DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = shape,
            OutputShape = shape,
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

    private Operation CreateOpWithTypeAndDtype(string type, RitterFramework.Core.DataType dtype)
    {
        return new TestOperation
        {
            Id = Guid.NewGuid().ToString(),
            Type = type,
            Name = $"{type}_{Guid.NewGuid():N}",
            DataType = dtype,
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
