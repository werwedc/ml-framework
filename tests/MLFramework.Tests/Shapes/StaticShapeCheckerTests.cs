using MLFramework.Core;
using MLFramework.Fusion;
using MLFramework.Shapes;
using MLFramework.Shapes.Inference;

namespace MLFramework.Tests.Shapes;

/// <summary>
/// Unit tests for the StaticShapeChecker class.
/// </summary>
[TestFixture]
public class StaticShapeCheckerTests
{
    private StaticShapeChecker _checker = null!;
    private ShapeInferenceEngine _inferenceEngine = null!;

    [SetUp]
    public void SetUp()
    {
        _inferenceEngine = new ShapeInferenceEngine();
        _checker = new StaticShapeChecker(_inferenceEngine);

        // Register a simple Add rule for testing
        _inferenceEngine.RegisterRule("Add", new AddInferenceRule());
        _inferenceEngine.RegisterRule("MatMul", new MatMulInferenceRule());
    }

    [Test]
    public void CheckOperation_WithNullOperation_ThrowsArgumentNullException()
    {
        // Arrange
        var inputs = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64))
        };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _checker.CheckOperation(null!, inputs));
    }

    [Test]
    public void CheckOperation_WithNullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var op = CreateTestOperation("Add");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _checker.CheckOperation(op, null!));
    }

    [Test]
    public void CheckOperation_WithUnregisteredOperation_ThrowsShapeMismatchException()
    {
        // Arrange
        var op = CreateTestOperation("UnknownOp");
        var inputs = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32))
        };

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckOperation(op, inputs));
        Assert.That(ex.Message, Does.Contain("UnknownOp"));
        Assert.That(ex.Message, Does.Contain("not registered"));
    }

    [Test]
    public void CheckOperation_WithValidInput_Passes()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var inputs = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64)),
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64))
        };

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckOperation(op, inputs));
    }

    [Test]
    public void CheckBroadcastCompatibility_WithCompatibleShapes_Passes()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64));
        var shapeB = new SymbolicShape(new SymbolicDimension("feat", 64));

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckBroadcastCompatibility(shapeA, shapeB));
    }

    [Test]
    public void CheckBroadcastCompatibility_WithCompatibleBroadcastShapes_Passes()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64));
        var shapeB = new SymbolicShape(new SymbolicDimension("batch", 1), new SymbolicDimension("feat", 64));

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckBroadcastCompatibility(shapeA, shapeB));
    }

    [Test]
    public void CheckBroadcastCompatibility_WithIncompatibleShapes_ThrowsShapeMismatchException()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64));
        var shapeB = new SymbolicShape(new SymbolicDimension("batch", 16), new SymbolicDimension("feat", 64));

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckBroadcastCompatibility(shapeA, shapeB));
        Assert.That(ex.Message, Does.Contain("Broadcast"));
        Assert.That(ex.Message, Does.Contain("not compatible"));
    }

    [Test]
    public void CheckBroadcastCompatibility_WithSymbolicShapes_PassesIfBoundsCompatible()
    {
        // Arrange
        var shapeA = new SymbolicShape(
            new SymbolicDimension("batch", null, 1, 256),
            new SymbolicDimension("feat", 64));
        var shapeB = new SymbolicShape(
            new SymbolicDimension("batch", null, 1, 128),
            new SymbolicDimension("feat", 64));

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckBroadcastCompatibility(shapeA, shapeB));
    }

    [Test]
    public void CheckReshapeValid_WithValidReshape_Passes()
    {
        // Arrange
        var from = new SymbolicShape(new SymbolicDimension("x", 12), new SymbolicDimension("y", 10));
        var to = new SymbolicShape(new SymbolicDimension("x", 120));

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckReshapeValid(from, to));
    }

    [Test]
    public void CheckReshapeValid_WithInvalidReshape_ThrowsShapeMismatchException()
    {
        // Arrange
        var from = new SymbolicShape(new SymbolicDimension("x", 12), new SymbolicDimension("y", 10));
        var to = new SymbolicShape(new SymbolicDimension("x", 50));

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckReshapeValid(from, to));
        Assert.That(ex.Message, Does.Contain("Reshape"));
        Assert.That(ex.Message, Does.Contain("element count"));
    }

    [Test]
    public void CheckReshapeValid_WithSymbolicBounds_PassesIfBoundsCompatible()
    {
        // Arrange
        var from = new SymbolicShape(
            new SymbolicDimension("x", null, 10, 20),
            new SymbolicDimension("y", null, 10, 20));
        var to = new SymbolicShape(
            new SymbolicDimension("z", null, 100, 400));

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckReshapeValid(from, to));
    }

    [Test]
    public void CheckReshapeValid_WithIncompatibleBounds_ThrowsShapeMismatchException()
    {
        // Arrange
        var from = new SymbolicShape(
            new SymbolicDimension("x", null, 100, 200),
            new SymbolicDimension("y", null, 100, 200));
        var to = new SymbolicShape(
            new SymbolicDimension("z", null, 50, 100));

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckReshapeValid(from, to));
        Assert.That(ex.Message, Does.Contain("Reshape"));
    }

    [Test]
    public void CheckMatMulCompatibility_WithValidMatMul_Passes()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("m", 10), new SymbolicDimension("n", 20));
        var shapeB = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("m", 20), new SymbolicDimension("n", 30));

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckMatMulCompatibility(shapeA, shapeB));
    }

    [Test]
    public void CheckMatMulCompatibility_WithInvalidInnerDimensions_ThrowsShapeMismatchException()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("m", 10), new SymbolicDimension("n", 20));
        var shapeB = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("m", 15), new SymbolicDimension("n", 30));

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckMatMulCompatibility(shapeA, shapeB));
        Assert.That(ex.Message, Does.Contain("MatMul"));
        Assert.That(ex.Message, Does.Contain("inner dimensions"));
    }

    [Test]
    public void CheckMatMulCompatibility_WithInsufficientRank_ThrowsShapeMismatchException()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("m", 10));
        var shapeB = new SymbolicShape(new SymbolicDimension("m", 10), new SymbolicDimension("n", 20));

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckMatMulCompatibility(shapeA, shapeB));
        Assert.That(ex.Message, Does.Contain("insufficient dimensions"));
    }

    [Test]
    public void CheckMatMulCompatibility_WithIncompatibleBatchDims_ThrowsShapeMismatchException()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("m", 10), new SymbolicDimension("n", 20));
        var shapeB = new SymbolicShape(new SymbolicDimension("batch", 16), new SymbolicDimension("m", 20), new SymbolicDimension("n", 30));

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckMatMulCompatibility(shapeA, shapeB));
        Assert.That(ex.Message, Does.Contain("Batch dimension"));
    }

    [Test]
    public void CheckRank_WithMatchingRank_Passes()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var inputs = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 10), new SymbolicDimension("y", 20))
        };

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckRank(op, inputs, 2));
    }

    [Test]
    public void CheckRank_WithMismatchingRank_ThrowsShapeMismatchException()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var inputs = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 10), new SymbolicDimension("y", 20))
        };

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckRank(op, inputs, 3));
        Assert.That(ex.Message, Does.Contain("Input 0"));
        Assert.That(ex.Message, Does.Contain("rank 2"));
        Assert.That(ex.Message, Does.Contain("expected 3"));
    }

    [Test]
    public void CheckDim_WithMatchingDimension_Passes()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var shape = new SymbolicShape(new SymbolicDimension("x", 10), new SymbolicDimension("y", 20));

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckDim(op, shape, 0, 10));
    }

    [Test]
    public void CheckDim_WithMismatchingDimension_ThrowsShapeMismatchException()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var shape = new SymbolicShape(new SymbolicDimension("x", 10), new SymbolicDimension("y", 20));

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckDim(op, shape, 0, 15));
        Assert.That(ex.Message, Does.Contain("Dimension 0"));
        Assert.That(ex.Message, Does.Contain("10"));
        Assert.That(ex.Message, Does.Contain("15"));
    }

    [Test]
    public void CheckDimRange_WithDimensionInRange_Passes()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var shape = new SymbolicShape(new SymbolicDimension("x", 15), new SymbolicDimension("y", 20));

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckDimRange(op, shape, 0, 10, 20));
    }

    [Test]
    public void CheckDimRange_WithDimensionOutOfRange_ThrowsShapeMismatchException()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var shape = new SymbolicShape(new SymbolicDimension("x", 25), new SymbolicDimension("y", 20));

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckDimRange(op, shape, 0, 10, 20));
        Assert.That(ex.Message, Does.Contain("Dimension 0"));
        Assert.That(ex.Message, Does.Contain("25"));
        Assert.That(ex.Message, Does.Contain("[10..20]"));
    }

    [Test]
    public void CheckSequence_WithValidSequence_Passes()
    {
        // Arrange
        var ops = new List<Operation>
        {
            CreateTestOperation("Add", new List<string> { "input1", "input2" }, new List<string> { "output1" }),
            CreateTestOperation("Add", new List<string> { "output1", "input3" }, new List<string> { "output2" })
        };

        var tensorShapes = new Dictionary<string, SymbolicShape>
        {
            ["input1"] = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64)),
            ["input2"] = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64)),
            ["input3"] = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64))
        };

        // Act & Assert
        Assert.DoesNotThrow(() => _checker.CheckSequence(ops, tensorShapes));
    }

    [Test]
    public void CheckSequence_WithMissingTensor_ThrowsShapeMismatchException()
    {
        // Arrange
        var ops = new List<Operation>
        {
            CreateTestOperation("Add", new List<string> { "missing_tensor", "input2" }, new List<string> { "output1" })
        };

        var tensorShapes = new Dictionary<string, SymbolicShape>
        {
            ["input2"] = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64))
        };

        // Act & Assert
        var ex = Assert.Throws<ShapeMismatchException>(() => _checker.CheckSequence(ops, tensorShapes));
        Assert.That(ex.Message, Does.Contain("missing_tensor"));
        Assert.That(ex.Message, Does.Contain("not found"));
    }

    #region Helper Classes and Methods

    private Operation CreateTestOperation(string opType, List<string>? inputs = null, List<string>? outputs = null)
    {
        return new TestOperation
        {
            Id = Guid.NewGuid().ToString(),
            Type = opType,
            Name = $"{opType}_test",
            DataType = DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = new TensorShape(new long[] { 32, 64 }),
            OutputShape = new TensorShape(new long[] { 32, 64 }),
            Inputs = inputs ?? new List<string> { "input1", "input2" },
            Outputs = outputs ?? new List<string> { "output1" },
            Attributes = new Dictionary<string, object>()
        };
    }

    private class TestOperation : Operation
    {
        // Empty implementation for testing
    }

    private class AddInferenceRule : IShapeInferenceRule
    {
        public bool CanInfer(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            return inputs.Count == 2 && inputs[0].Rank == inputs[1].Rank;
        }

        public List<SymbolicShape> Infer(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            return new List<SymbolicShape> { inputs[0] }; // Add preserves shape
        }
    }

    private class MatMulInferenceRule : IShapeInferenceRule
    {
        public bool CanInfer(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            return inputs.Count == 2 && inputs[0].Rank >= 2 && inputs[1].Rank >= 2;
        }

        public List<SymbolicShape> Infer(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            var a = inputs[0];
            var b = inputs[1];

            // Result shape: [..., M, P] where a is [..., M, N] and b is [..., N, P]
            var maxRank = Math.Max(a.Rank, b.Rank);
            var resultDims = new List<SymbolicDimension>();

            for (int i = 0; i < maxRank - 2; i++)
            {
                var dimA = a.Rank >= maxRank ? a.GetDimension(i) : null;
                var dimB = b.Rank >= maxRank ? b.GetDimension(i) : null;
                resultDims.Add(dimA ?? dimB ?? new SymbolicDimension($"dim_{i}"));
            }

            resultDims.Add(a.GetDimension(-2)); // M
            resultDims.Add(b.GetDimension(-1)); // P

            return new List<SymbolicShape> { new SymbolicShape(resultDims) };
        }
    }

    #endregion
}
