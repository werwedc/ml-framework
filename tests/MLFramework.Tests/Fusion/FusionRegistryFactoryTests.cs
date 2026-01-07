using NUnit.Framework;
using MLFramework.Fusion;

namespace MLFramework.Tests.Fusion;

/// <summary>
/// Tests for FusionRegistryFactory
/// </summary>
[TestFixture]
public class FusionRegistryFactoryTests
{
    [Test]
    public void CreateDefault_ReturnsRegistryWithPatterns()
    {
        var registry = FusionRegistryFactory.CreateDefault();

        Assert.That(registry, Is.Not.Null);
        Assert.That(registry.GetFusibleOperations().Count, Is.GreaterThan(0));
        Assert.That(registry.GetPattern("ElementWiseChain"), Is.Not.Null);
        Assert.That(registry.GetPattern("ConvActivation"), Is.Not.Null);
    }

    [Test]
    public void CreateEmpty_ReturnsRegistryWithoutPatterns()
    {
        var registry = FusionRegistryFactory.CreateEmpty();

        Assert.That(registry, Is.Not.Null);
        Assert.That(registry.GetFusibleOperations().Count, Is.EqualTo(0));
    }

    [Test]
    public void CreateWithCustomPatterns_AppliesConfiguration()
    {
        var registry = FusionRegistryFactory.CreateWithCustomPatterns(r =>
        {
            r.RegisterFusibleOperation("CustomOp", new FusibleOpConstraints
            {
                RequiredLayout = TensorLayout.Any,
                SupportedDataTypes = new HashSet<MLFramework.Core.DataType>
                {
                    MLFramework.Core.DataType.Float32
                }
            });
        });

        var fusibleOps = registry.GetFusibleOperations();

        Assert.That(fusibleOps, Does.Contain("CustomOp"));
    }

    [Test]
    public void CreateWithCustomPatterns_NullConfigure_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            FusionRegistryFactory.CreateWithCustomPatterns(null!));
    }

    [Test]
    public void CreateWithCustomPatterns_IncludesDefaults()
    {
        var registry = FusionRegistryFactory.CreateWithCustomPatterns(r =>
        {
            r.RegisterFusibleOperation("CustomOp", new FusibleOpConstraints
            {
                RequiredLayout = TensorLayout.Any,
                SupportedDataTypes = new HashSet<MLFramework.Core.DataType>
                {
                    MLFramework.Core.DataType.Float32
                }
            });
        });

        // Should include both default and custom operations
        Assert.That(registry.GetFusibleOperations(), Does.Contain("Add"));
        Assert.That(registry.GetFusibleOperations(), Does.Contain("CustomOp"));
        Assert.That(registry.GetPattern("ElementWiseChain"), Is.Not.Null);
    }
}
