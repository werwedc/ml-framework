using NUnit.Framework;
using MLFramework.Fusion;
using MLFramework.Core;

namespace MLFramework.Tests.Fusion;

/// <summary>
/// Tests for Fusion API attributes and options
/// </summary>
[TestFixture]
public class FusionApiTests
{
    [Test]
    public void FusibleAttribute_ReadsProperties()
    {
        var attr = new FusibleAttribute
        {
            MaxOperations = 5,
            Strategy = FusionStrategy.Fold,
            Priority = 10
        };

        Assert.AreEqual(5, attr.MaxOperations);
        Assert.AreEqual(FusionStrategy.Fold, attr.Strategy);
        Assert.AreEqual(10, attr.Priority);
    }

    [Test]
    public void FusibleAttribute_DefaultValues()
    {
        var attr = new FusibleAttribute();

        Assert.AreEqual(10, attr.MaxOperations);
        Assert.AreEqual(FusionStrategy.Merge, attr.Strategy);
        Assert.AreEqual(0, attr.Priority);
        Assert.IsNull(attr.Pattern);
    }

    [Test]
    public void FusibleAttribute_PatternProperty()
    {
        var attr = new FusibleAttribute
        {
            Pattern = "CustomPattern"
        };

        Assert.AreEqual("CustomPattern", attr.Pattern);
    }

    [Test]
    public void NoFusionAttribute_DefaultReason()
    {
        var attr = new NoFusionAttribute();

        Assert.AreEqual("Explicitly marked as non-fusible", attr.Reason);
    }

    [Test]
    public void NoFusionAttribute_CustomReason()
    {
        var attr = new NoFusionAttribute("Contains data-dependent control flow");

        Assert.AreEqual("Contains data-dependent control flow", attr.Reason);
    }

    [Test]
    public void NoFusionAttribute_SetReason()
    {
        var attr = new NoFusionAttribute();
        attr.Reason = "Custom reason";

        Assert.AreEqual("Custom reason", attr.Reason);
    }

    [Test]
    public void GraphOptions_DefaultValues()
    {
        Assert.IsTrue(GraphOptions.EnableFusion);
        Assert.AreEqual(10, GraphOptions.MaxFusionOps);
        Assert.AreEqual(FusionBackend.Triton, GraphOptions.FusionBackend);
        Assert.AreEqual(50, GraphOptions.MinBenefitScore);
        Assert.AreEqual(FusionAggressiveness.Medium, GraphOptions.Aggressiveness);
        Assert.IsTrue(GraphOptions.EnableAutomaticFusion);
        Assert.IsTrue(GraphOptions.EnableHintedFusion);
        Assert.IsTrue(GraphOptions.EnableAutotuning);
        Assert.IsTrue(GraphOptions.EnableBatchNormFolding);
        Assert.IsTrue(GraphOptions.EnableConvActivationFusion);
        Assert.IsTrue(GraphOptions.EnableElementWiseFusion);
        Assert.IsNull(GraphOptions.TuningCacheDirectory);
    }

    [Test]
    public void GraphOptions_ModifyValues()
    {
        var originalMaxOps = GraphOptions.MaxFusionOps;

        GraphOptions.MaxFusionOps = 20;
        GraphOptions.EnableFusion = false;
        GraphOptions.FusionBackend = FusionBackend.XLA;

        Assert.AreEqual(20, GraphOptions.MaxFusionOps);
        Assert.IsFalse(GraphOptions.EnableFusion);
        Assert.AreEqual(FusionBackend.XLA, GraphOptions.FusionBackend);

        // Restore
        GraphOptions.MaxFusionOps = originalMaxOps;
        GraphOptions.EnableFusion = true;
        GraphOptions.FusionBackend = FusionBackend.Triton;
    }

    [Test]
    public void GraphOptions_ResetsToDefaults()
    {
        GraphOptions.MaxFusionOps = 20;
        GraphOptions.EnableFusion = false;
        GraphOptions.FusionBackend = FusionBackend.Custom;
        GraphOptions.MinBenefitScore = 100;
        GraphOptions.Aggressiveness = FusionAggressiveness.Aggressive;

        GraphOptions.ResetToDefaults();

        Assert.AreEqual(10, GraphOptions.MaxFusionOps);
        Assert.IsTrue(GraphOptions.EnableFusion);
        Assert.AreEqual(FusionBackend.Triton, GraphOptions.FusionBackend);
        Assert.AreEqual(50, GraphOptions.MinBenefitScore);
        Assert.AreEqual(FusionAggressiveness.Medium, GraphOptions.Aggressiveness);
    }

    [Test]
    public void GraphOptions_TuningCacheDirectory()
    {
        var originalCacheDir = GraphOptions.TuningCacheDirectory;

        GraphOptions.TuningCacheDirectory = "/tmp/fusion_cache";
        Assert.AreEqual("/tmp/fusion_cache", GraphOptions.TuningCacheDirectory);

        GraphOptions.TuningCacheDirectory = originalCacheDir;
    }

    [Test]
    public void FusionContext_DisablesFusion()
    {
        var originalEnableFusion = GraphOptions.EnableFusion;

        using (FusionContext.DisableFusion())
        {
            Assert.IsFalse(GraphOptions.EnableFusion);
        }

        Assert.AreEqual(originalEnableFusion, GraphOptions.EnableFusion);
    }

    [Test]
    public void FusionContext_RestoresOptionsOnDispose()
    {
        var originalMaxOps = GraphOptions.MaxFusionOps;

        using (FusionContext.WithOptions(opts => opts.MaxFusionOps = 20))
        {
            Assert.AreEqual(20, GraphOptions.MaxFusionOps);
        }

        Assert.AreEqual(originalMaxOps, GraphOptions.MaxFusionOps);
    }

    [Test]
    public void FusionContext_MultipleOptions()
    {
        var originalMaxOps = GraphOptions.MaxFusionOps;
        var originalEnableFusion = GraphOptions.EnableFusion;

        using (FusionContext.WithOptions(opts =>
        {
            opts.MaxFusionOps = 20;
            opts.EnableFusion = false;
            opts.FusionBackend = FusionBackend.Custom;
        }))
        {
            Assert.AreEqual(20, GraphOptions.MaxFusionOps);
            Assert.IsFalse(GraphOptions.EnableFusion);
            Assert.AreEqual(FusionBackend.Custom, GraphOptions.FusionBackend);
        }

        Assert.AreEqual(originalMaxOps, GraphOptions.MaxFusionOps);
        Assert.AreEqual(originalEnableFusion, GraphOptions.EnableFusion);
    }

    [Test]
    public void FusionContext_NoRestoreOnDispose()
    {
        var originalMaxOps = GraphOptions.MaxFusionOps;

        using (new FusionContext(new FusionOptions { MaxFusionOps = 20 }, restoreOnDispose: false))
        {
            Assert.AreEqual(20, GraphOptions.MaxFusionOps);
        }

        // Options should remain changed since we set restoreOnDispose to false
        Assert.AreEqual(20, GraphOptions.MaxFusionOps);

        // Restore manually
        GraphOptions.MaxFusionOps = originalMaxOps;
    }

    [Test]
    public void FusionStatistics_CollectsCorrectly()
    {
        var collector = new FusionStatisticsCollector();

        collector.RecordOperation(CreateTestOperation("Add"));
        collector.RecordOperation(CreateTestOperation("Mul"));
        collector.RecordFusedGroup(
            new[] { CreateTestOperation("ReLU"), CreateTestOperation("Sigmoid") },
            FusionPatternType.ElementWise);
        collector.RecordRejection("Layout mismatch");

        var stats = collector.GetCurrentStatistics();

        Assert.AreEqual(4, stats.TotalOperations);
        Assert.AreEqual(2, stats.FusedOperations);
        Assert.AreEqual(1, stats.FusedGroups);
        Assert.AreEqual(1, stats.RejectedFusions);
        Assert.AreEqual(50.0, stats.FusionPercentage, 0.01);
        Assert.AreEqual(2.0, stats.AverageOperationsPerFusedGroup, 0.01);
        Assert.IsTrue(stats.PatternCounts.ContainsKey(FusionPatternType.ElementWise));
        Assert.AreEqual(1, stats.PatternCounts[FusionPatternType.ElementWise]);
        Assert.IsTrue(stats.RejectionReasons.ContainsKey("Layout mismatch"));
        Assert.AreEqual(1, stats.RejectionReasons["Layout mismatch"]);
    }

    [Test]
    public void FusionStatistics_Reset()
    {
        var collector = new FusionStatisticsCollector();

        collector.RecordOperation(CreateTestOperation("Add"));
        collector.RecordFusedGroup(
            new[] { CreateTestOperation("ReLU") },
            FusionPatternType.ElementWise);

        Assert.AreEqual(2, collector.GetCurrentStatistics().TotalOperations);

        collector.Reset();

        Assert.AreEqual(0, collector.GetCurrentStatistics().TotalOperations);
        Assert.AreEqual(0, collector.GetCurrentStatistics().FusedGroups);
    }

    [Test]
    public void FusionStatistics_ThreadSafety()
    {
        var collector = new FusionStatisticsCollector();
        const int threadCount = 10;
        const int opsPerThread = 100;

        Parallel.For(0, threadCount, i =>
        {
            for (int j = 0; j < opsPerThread; j++)
            {
                collector.RecordOperation(CreateTestOperation($"Op_{i}_{j}"));
            }
        });

        var stats = collector.GetCurrentStatistics();
        Assert.AreEqual(threadCount * opsPerThread, stats.TotalOperations);
    }

    [Test]
    public void FusionStatistics_PatternCounts()
    {
        var collector = new FusionStatisticsCollector();

        collector.RecordFusedGroup(
            new[] { CreateTestOperation("Conv") },
            FusionPatternType.ConvActivation);
        collector.RecordFusedGroup(
            new[] { CreateTestOperation("Add"), CreateTestOperation("ReLU") },
            FusionPatternType.ElementWise);
        collector.RecordFusedGroup(
            new[] { CreateTestOperation("Add"), CreateTestOperation("Sigmoid") },
            FusionPatternType.ElementWise);

        var stats = collector.GetCurrentStatistics();

        Assert.AreEqual(3, stats.FusedGroups);
        Assert.AreEqual(1, stats.PatternCounts[FusionPatternType.ConvActivation]);
        Assert.AreEqual(2, stats.PatternCounts[FusionPatternType.ElementWise]);
    }

    [Test]
    public void FusionStatistics_RejectionReasons()
    {
        var collector = new FusionStatisticsCollector();

        collector.RecordRejection("Layout mismatch");
        collector.RecordRejection("Layout mismatch");
        collector.RecordRejection("Side effect");

        var stats = collector.GetCurrentStatistics();

        Assert.AreEqual(3, stats.RejectedFusions);
        Assert.AreEqual(2, stats.RejectionReasons["Layout mismatch"]);
        Assert.AreEqual(1, stats.RejectionReasons["Side effect"]);
    }

    [Test]
    public void FusionStatistics_LogFusionDecisions_DoesNotThrow()
    {
        var collector = new FusionStatisticsCollector();

        collector.RecordOperation(CreateTestOperation("Add"));
        collector.RecordFusedGroup(
            new[] { CreateTestOperation("ReLU") },
            FusionPatternType.ElementWise);
        collector.RecordRejection("Layout mismatch");

        Assert.DoesNotThrow(() => collector.LogFusionDecisions());
    }

    [Test]
    public void FusionOptions_RecordType()
    {
        var options = new FusionOptions
        {
            EnableFusion = false,
            MaxFusionOps = 20,
            FusionBackend = FusionBackend.Custom,
            MinBenefitScore = 80,
            Aggressiveness = FusionAggressiveness.Aggressive
        };

        Assert.IsFalse(options.EnableFusion);
        Assert.AreEqual(20, options.MaxFusionOps);
        Assert.AreEqual(FusionBackend.Custom, options.FusionBackend);
        Assert.AreEqual(80, options.MinBenefitScore);
        Assert.AreEqual(FusionAggressiveness.Aggressive, options.Aggressiveness);
    }

    [Test]
    public void FusionBackend_EnumValues()
    {
        Assert.AreEqual(4, System.Enum.GetValues<FusionBackend>().Length);
        Assert.IsTrue(System.Enum.IsDefined(typeof(FusionBackend), FusionBackend.Triton));
        Assert.IsTrue(System.Enum.IsDefined(typeof(FusionBackend), FusionBackend.XLA));
        Assert.IsTrue(System.Enum.IsDefined(typeof(FusionBackend), FusionBackend.Custom));
        Assert.IsTrue(System.Enum.IsDefined(typeof(FusionBackend), FusionBackend.None));
    }

    [Test]
    public void FusionAggressiveness_EnumValues()
    {
        Assert.AreEqual(3, System.Enum.GetValues<FusionAggressiveness>().Length);
        Assert.IsTrue(System.Enum.IsDefined(typeof(FusionAggressiveness), FusionAggressiveness.Conservative));
        Assert.IsTrue(System.Enum.IsDefined(typeof(FusionAggressiveness), FusionAggressiveness.Medium));
        Assert.IsTrue(System.Enum.IsDefined(typeof(FusionAggressiveness), FusionAggressiveness.Aggressive));
    }

    /// <summary>
    /// Helper method to create a test operation
    /// </summary>
    private Operation CreateTestOperation(string type)
    {
        return new TestOperation
        {
            Id = $"{type}_{Guid.NewGuid()}",
            Type = type,
            Name = type,
            DataType = DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = new TensorShape { Batch = 1, Height = 32, Width = 32, Channels = 3 },
            OutputShape = new TensorShape { Batch = 1, Height = 32, Width = 32, Channels = 3 },
            Inputs = Array.Empty<string>(),
            Outputs = new[] { "output" },
            Attributes = new Dictionary<string, object>()
        };
    }

    /// <summary>
    /// Test operation implementation for testing
    /// </summary>
    private record TestOperation : Operation;
}
