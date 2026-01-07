using NUnit.Framework;
using MLFramework.Core;
using MLFramework.Fusion;

namespace MLFramework.Tests.Fusion;

/// <summary>
/// Tests for DefaultFusionRegistry
/// </summary>
[TestFixture]
public class DefaultFusionRegistryTests
{
    [Test]
    public void RegisterOperation_Retrievable()
    {
        var registry = new DefaultFusionRegistry(skipDefaults: true);
        registry.RegisterFusibleOperation("CustomOp", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.NCHW,
            SupportedDataTypes = new HashSet<DataType> { DataType.Float32 }
        });

        var fusibleOps = registry.GetFusibleOperations();

        Assert.That(fusibleOps, Does.Contain("CustomOp"));
    }

    [Test]
    public void RegisterOperation_OverwritesExisting()
    {
        var registry = new DefaultFusionRegistry();
        var originalConstraints = registry.GetPattern("ElementWiseChain");

        registry.RegisterFusibleOperation("Add", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> { DataType.Float16 }
        });

        var fusibleOps = registry.GetFusibleOperations();
        Assert.That(fusibleOps, Does.Contain("Add"));
    }

    [Test]
    public void RegisterPattern_Retrievable()
    {
        var registry = new DefaultFusionRegistry(skipDefaults: true);
        var pattern = new FusionPatternDefinition
        {
            Name = "TestPattern",
            OpTypeSequence = new[] { "TestOp1", "TestOp2" },
            MatchStrategy = ops => true,
            Strategy = FusionStrategy.Merge,
            Priority = 5
        };

        registry.RegisterFusionPattern("TestPattern", pattern);
        var retrieved = registry.GetPattern("TestPattern");

        Assert.That(retrieved, Is.Not.Null);
        Assert.That(retrieved!.Name, Is.EqualTo("TestPattern"));
    }

    [Test]
    public void GetPattern_NotFound_ReturnsNull()
    {
        var registry = new DefaultFusionRegistry();
        var pattern = registry.GetPattern("NonExistentPattern");

        Assert.That(pattern, Is.Null);
    }

    [Test]
    public void FindPattern_MatchesConvReLU()
    {
        var registry = new DefaultFusionRegistry();
        var ops = new[]
        {
            OperationTestHelper.CreateConvOp(),
            OperationTestHelper.CreateReluOp()
        };

        var matches = registry.FindMatches(ops);

        Assert.That(matches.Count, Is.EqualTo(1));
        Assert.That(matches[0].Pattern.Name, Is.EqualTo("ConvActivation"));
        Assert.That(matches[0].MatchedOperations.Count, Is.EqualTo(2));
    }

    [Test]
    public void FindPattern_NoMatch_ReturnsEmptyList()
    {
        var registry = new DefaultFusionRegistry(skipDefaults: true);
        var ops = new[]
        {
            OperationTestHelper.CreateOperation("UnknownOp1"),
            OperationTestHelper.CreateOperation("UnknownOp2")
        };

        var matches = registry.FindMatches(ops);

        Assert.That(matches.Count, Is.EqualTo(0));
    }

    [Test]
    public void FindPattern_ElementWiseChain_ReturnsMatch()
    {
        var registry = new DefaultFusionRegistry();
        var ops = new[]
        {
            OperationTestHelper.CreateAddOp(),
            OperationTestHelper.CreateOperation("Mul"),
            OperationTestHelper.CreateReluOp()
        };

        var matches = registry.FindMatches(ops);

        Assert.That(matches.Count, Is.EqualTo(1));
        Assert.That(matches[0].Pattern.Name, Is.EqualTo("ElementWiseChain"));
    }

    [Test]
    public void FindPattern_LinearReLU_ReturnsMatch()
    {
        var registry = new DefaultFusionRegistry();
        var ops = new[]
        {
            OperationTestHelper.CreateLinearOp(),
            OperationTestHelper.CreateReluOp()
        };

        var matches = registry.FindMatches(ops);

        Assert.That(matches.Count, Is.EqualTo(1));
        Assert.That(matches[0].Pattern.Name, Is.EqualTo("LinearActivation"));
    }

    [Test]
    public void FindPattern_ConvBatchNorm_InferenceMode_ReturnsMatch()
    {
        var registry = new DefaultFusionRegistry();
        var ops = new[]
        {
            OperationTestHelper.CreateConvOp(),
            OperationTestHelper.CreateBatchNormOp(training: false)
        };

        var matches = registry.FindMatches(ops);

        Assert.That(matches.Count, Is.EqualTo(1));
        Assert.That(matches[0].Pattern.Name, Is.EqualTo("ConvBatchNorm"));
    }

    [Test]
    public void FindPattern_ConvBatchNorm_TrainingMode_ReturnsNoMatch()
    {
        var registry = new DefaultFusionRegistry();
        var ops = new[]
        {
            OperationTestHelper.CreateConvOp(),
            OperationTestHelper.CreateBatchNormOp(training: true)
        };

        var matches = registry.FindMatches(ops);

        Assert.That(matches.Count, Is.EqualTo(0));
    }

    [Test]
    public void FindPattern_MultiplePatterns_SortedByPriority()
    {
        var registry = new DefaultFusionRegistry(skipDefaults: true);

        // Register two patterns with different priorities
        registry.RegisterFusionPattern("LowPriority", new FusionPatternDefinition
        {
            Name = "LowPriority",
            OpTypeSequence = new[] { "Add", "ReLU" },
            MatchStrategy = ops => true,
            Priority = 5
        });

        registry.RegisterFusionPattern("HighPriority", new FusionPatternDefinition
        {
            Name = "HighPriority",
            OpTypeSequence = new[] { "Add", "ReLU" },
            MatchStrategy = ops => true,
            Priority = 20
        });

        var ops = new[]
        {
            OperationTestHelper.CreateAddOp(),
            OperationTestHelper.CreateReluOp()
        };

        var matches = registry.FindMatches(ops);

        Assert.That(matches.Count, Is.EqualTo(2));
        Assert.That(matches[0].Pattern.Name, Is.EqualTo("HighPriority"));
        Assert.That(matches[1].Pattern.Name, Is.EqualTo("LowPriority"));
    }

    [Test]
    public void FindPattern_MatchScore_IncludesOperationCount()
    {
        var registry = new DefaultFusionRegistry();
        var ops = new[]
        {
            OperationTestHelper.CreateAddOp(),
            OperationTestHelper.CreateOperation("Mul"),
            OperationTestHelper.CreateReluOp(),
            OperationTestHelper.CreateOperation("Sigmoid")
        };

        var matches = registry.FindMatches(ops);

        Assert.That(matches.Count, Is.EqualTo(1));
        // Score = priority(10) * 100 + operations(4) * 10 = 1000 + 40 = 1040
        Assert.That(matches[0].MatchScore, Is.GreaterThan(1000));
    }

    [Test]
    public void GetFusibleOperations_DefaultRegistry_ReturnsAllDefaultOps()
    {
        var registry = new DefaultFusionRegistry();
        var fusibleOps = registry.GetFusibleOperations();

        Assert.That(fusibleOps, Does.Contain("Add"));
        Assert.That(fusibleOps, Does.Contain("Mul"));
        Assert.That(fusibleOps, Does.Contain("ReLU"));
        Assert.That(fusibleOps, Does.Contain("Sigmoid"));
        Assert.That(fusibleOps, Does.Contain("Tanh"));
    }

    [Test]
    public void RegisterPattern_NullOperationType_ThrowsArgumentNullException()
    {
        var registry = new DefaultFusionRegistry();

        Assert.Throws<ArgumentNullException>(() =>
            registry.RegisterFusibleOperation(null!, new FusibleOpConstraints
            {
                RequiredLayout = TensorLayout.Any,
                SupportedDataTypes = new HashSet<DataType> { DataType.Float32 }
            }));
    }

    [Test]
    public void RegisterPattern_NullPattern_ThrowsArgumentNullException()
    {
        var registry = new DefaultFusionRegistry();

        Assert.Throws<ArgumentNullException>(() =>
            registry.RegisterFusionPattern("Test", null!));
    }

    [Test]
    public void FindMatches_NullOperations_ThrowsArgumentNullException()
    {
        var registry = new DefaultFusionRegistry();

        Assert.Throws<ArgumentNullException>(() =>
            registry.FindMatches(null!));
    }
}
