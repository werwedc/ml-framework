using NUnit.Framework;
using MLFramework.Fusion;

namespace MLFramework.Tests.Fusion;

/// <summary>
/// Unit tests for FusionValidator
/// </summary>
[TestFixture]
public class FusionValidatorTests
{
    private FusionValidator _validator = null!;
    private IFusionRegistry _registry = null!;
    private FusionEngine _engine = null!;

    [SetUp]
    public void Setup()
    {
        _validator = new FusionValidator();
        _registry = FusionRegistryFactory.CreateDefaultRegistry();
        _engine = new FusionEngine(_registry);
    }

    [Test]
    public void ValidateFusion_ValidFusion_PassesValidation()
    {
        // Arrange
        var operations = CreateOperationChain(new[] { "Add", "ReLU" });
        var pattern = _registry.GetPattern("ElementWiseChain")!;
        var fusedOp = _engine.FuseOperations(operations, pattern);
        var originalGraph = CreateSimpleGraph();

        // Act
        var result = _validator.ValidateFusion(fusedOp, originalGraph);

        // Assert
        Assert.IsTrue(result.IsValid);
        Assert.IsEmpty(result.Errors);
    }

    [Test]
    public void ValidateFusion_ShapeMismatch_FailsValidation()
    {
        // Arrange
        var operations = CreateOperationChain(new[] { "Add", "ReLU" });
        var pattern = _registry.GetPattern("ElementWiseChain")!;
        var fusedOp = _engine.FuseOperations(operations, pattern);

        // Manually create shape mismatch
        fusedOp = fusedOp with
        {
            InputShape = new TensorShape { Dimensions = new[] { 100, 200 } }
        };

        var originalGraph = CreateSimpleGraph();

        // Act
        var result = _validator.ValidateFusion(fusedOp, originalGraph);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.IsNotEmpty(result.Errors);
        Assert.IsTrue(result.Errors.Any(e => e.Contains("Shape preservation")));
    }

    [Test]
    public void ValidateFusion_NumericallyUnstableOps_GeneratesWarnings()
    {
        // Arrange
        var operations = CreateOperationChain(new[] { "Div", "Log" });
        var pattern = _registry.GetPattern("ElementWiseChain")!;
        var fusedOp = _engine.FuseOperations(operations, pattern);
        var originalGraph = CreateSimpleGraph();

        // Act
        var result = _validator.ValidateFusion(fusedOp, originalGraph);

        // Assert
        Assert.IsNotEmpty(result.Warnings);
        Assert.IsTrue(result.Warnings.Any(w =>
            w.Contains("Division") || w.Contains("Logarithm")));
    }

    [Test]
    public void ValidateFusion_ExcessiveSharedMemory_GeneratesWarning()
    {
        // Arrange
        var operations = CreateOperationChain(new[] { "Add", "Mul" });
        var pattern = _registry.GetPattern("ElementWiseChain")!;
        var fusedOp = _engine.FuseOperations(operations, pattern);

        // Modify IR to have excessive shared memory
        var ir = fusedOp.IntermediateRepresentation with
        {
            MemoryLayout = fusedOp.IntermediateRepresentation.MemoryLayout with
            {
                SharedMemoryBytes = 60 * 1024 // 60KB, exceeds 48KB limit
            }
        };

        fusedOp = fusedOp with { IntermediateRepresentation = ir };

        var originalGraph = CreateSimpleGraph();

        // Act
        var result = _validator.ValidateFusion(fusedOp, originalGraph);

        // Assert
        Assert.IsTrue(result.Warnings.Any(w => w.Contains("Shared memory")));
    }

    [Test]
    public void ValidatePatternApplicability_ValidPattern_ReturnsTrue()
    {
        // Arrange
        var operations = CreateOperationChain(new[] { "Add", "Mul", "ReLU" });
        var pattern = new FusionPatternDefinition
        {
            Name = "TestPattern",
            Strategy = FusionStrategy.Merge,
            MinOperations = 1,
            MaxOperations = 5,
            RequiredOperations = new[] { "Add", "Mul", "ReLU" }
        };

        // Act
        var result = _validator.ValidatePatternApplicability(operations, pattern);

        // Assert
        Assert.IsTrue(result);
    }

    [Test]
    public void ValidatePatternApplicability_TooManyOperations_ReturnsFalse()
    {
        // Arrange
        var operations = CreateOperationChain(new[] { "Add", "Mul", "ReLU" });
        var pattern = new FusionPatternDefinition
        {
            Name = "TestPattern",
            Strategy = FusionStrategy.Merge,
            MinOperations = 1,
            MaxOperations = 2, // Less than operations count
            RequiredOperations = new[] { "Add", "Mul", "ReLU" }
        };

        // Act
        var result = _validator.ValidatePatternApplicability(operations, pattern);

        // Assert
        Assert.IsFalse(result);
    }

    [Test]
    public void ValidatePatternApplicability_MissingRequiredOp_ReturnsFalse()
    {
        // Arrange
        var operations = CreateOperationChain(new[] { "Add", "Mul" });
        var pattern = new FusionPatternDefinition
        {
            Name = "TestPattern",
            Strategy = FusionStrategy.Merge,
            MinOperations = 1,
            MaxOperations = 5,
            RequiredOperations = new[] { "Add", "Mul", "ReLU" } // ReLU missing
        };

        // Act
        var result = _validator.ValidatePatternApplicability(operations, pattern);

        // Assert
        Assert.IsFalse(result);
    }

    private ComputationalGraph CreateSimpleGraph()
    {
        var operations = CreateOperationChain(new[] { "Add", "Mul", "ReLU" });

        return new ComputationalGraph
        {
            Id = "test_graph",
            Operations = operations,
            DependencyGraph = new DependencyGraph()
        };
    }

    private List<Operation> CreateOperationChain(string[] opTypes)
    {
        var operations = new List<Operation>();
        string currentOutput = "input";

        for (int i = 0; i < opTypes.Length; i++)
        {
            var opType = opTypes[i];
            var op = new TestOperation
            {
                Id = $"op_{i}",
                Type = opType,
                Name = $"{opType}_{i}",
                DataType = DataType.Float32,
                Layout = TensorLayout.Any,
                InputShape = i == 0
                    ? new TensorShape { Dimensions = new[] { 10, 20, 30 } }
                    : operations[i - 1].OutputShape,
                OutputShape = new TensorShape { Dimensions = new[] { 10, 20, 30 } },
                Inputs = new[] { currentOutput },
                Outputs = new[] { $"output_{i}" },
                Attributes = new Dictionary<string, object>()
            };

            operations.Add(op);
            currentOutput = $"output_{i}";
        }

        return operations;
    }

    private class TestOperation : Operation
    {
        public required string Id { get; init; }
        public required string Type { get; init; }
        public required string Name { get; init; }
        public required DataType DataType { get; init; }
        public required TensorLayout Layout { get; init; }
        public required TensorShape InputShape { get; init; }
        public required TensorShape OutputShape { get; init; }
        public required IReadOnlyList<string> Inputs { get; init; }
        public required IReadOnlyList<string> Outputs { get; init; }
        public required IReadOnlyDictionary<string, object> Attributes { get; init; }
    }
}
