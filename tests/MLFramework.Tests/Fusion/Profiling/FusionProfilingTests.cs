using NUnit.Framework;
using MLFramework.Fusion;
using FusionBackends = MLFramework.Fusion.Backends;
using MLFramework.Fusion.Profiling;
using MLFramework.Core;

namespace MLFramework.Tests.Fusion.Profiling;

/// <summary>
/// Unit tests for Fusion Profiling
/// </summary>
[TestFixture]
public class FusionProfilingTests
{
    private FusionProfiler _profiler = null!;
    private KernelTimingInstrument _timingInstrument = null!;
    private PerformanceReportGenerator _reportGenerator = null!;
    private IFusionDecisionLogger _consoleLogger = null!;

    [SetUp]
    public void Setup()
    {
        _profiler = new FusionProfiler();
        _timingInstrument = new KernelTimingInstrument();
        _reportGenerator = new PerformanceReportGenerator();
        _consoleLogger = new ConsoleFusionDecisionLogger(verbose: false);
    }

    [Test]
    public void Profiler_RecordDecision_Retrievable()
    {
        // Arrange
        var decision = new FusionDecision
        {
            OperationChain = "Add -> Mul",
            Fused = true,
            PatternType = FusionPatternType.ElementWise,
            RejectionReason = null,
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        // Act
        _profiler.RecordDecision(decision);
        var report = _profiler.GetReport();

        // Assert
        Assert.AreEqual(1, report.Decisions.Count);
        Assert.IsTrue(report.Decisions[0].Fused);
    }

    [Test]
    public void Profiler_RecordMultipleDecisions_AllRetrievable()
    {
        // Arrange
        var decision1 = new FusionDecision
        {
            OperationChain = "Add -> Mul",
            Fused = true,
            PatternType = FusionPatternType.ElementWise,
            RejectionReason = null,
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        var decision2 = new FusionDecision
        {
            OperationChain = "Conv2D -> ReLU",
            Fused = true,
            PatternType = FusionPatternType.ConvActivation,
            RejectionReason = null,
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        var decision3 = new FusionDecision
        {
            OperationChain = "Div -> Sub",
            Fused = false,
            PatternType = FusionPatternType.ElementWise,
            RejectionReason = "Shape mismatch",
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        // Act
        _profiler.RecordDecision(decision1);
        _profiler.RecordDecision(decision2);
        _profiler.RecordDecision(decision3);
        var report = _profiler.GetReport();

        // Assert
        Assert.AreEqual(3, report.Decisions.Count);
        Assert.AreEqual(2, report.Summary.SuccessfulFusions);
        Assert.AreEqual(1, report.Summary.FailedFusions);
    }

    [Test]
    public void Profiler_StartProfiling_SessionRecordsKernelTime()
    {
        // Arrange
        var fusedOp = CreateTestFusedOperation();
        var session = _profiler.StartProfiling(fusedOp);

        // Act
        System.Threading.Thread.Sleep(10); // Simulate some work
        session.Dispose();

        // Assert
        var report = _profiler.GetReport();
        Assert.AreEqual(1, report.KernelExecutions.Count);
        Assert.Greater(report.KernelExecutions[0].DurationMs, 0);
    }

    [Test]
    public void Profiler_ComputeSummary_CalculatesCorrectMetrics()
    {
        // Arrange
        var decision1 = new FusionDecision
        {
            OperationChain = "Add -> Mul -> ReLU",
            Fused = true,
            PatternType = FusionPatternType.ElementWise,
            RejectionReason = null,
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        var decision2 = new FusionDecision
        {
            OperationChain = "Div -> Sub",
            Fused = false,
            PatternType = FusionPatternType.ElementWise,
            RejectionReason = "Too few ops",
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        _profiler.RecordDecision(decision1);
        _profiler.RecordDecision(decision2);

        // Act
        var report = _profiler.GetReport();

        // Assert
        Assert.AreEqual(5, report.Summary.TotalOperations); // 3 + 2
        Assert.AreEqual(3, report.Summary.FusedOperations);
        Assert.AreEqual(1, report.Summary.FusedGroups);
        Assert.AreEqual(60.0, report.Summary.FusionRate, 0.01); // 3/5 * 100
    }

    [Test]
    public void KernelTimingInstrument_ComputesStatistics()
    {
        // Arrange
        var config = CreateLaunchConfig();
        _timingInstrument.RecordKernelLaunch("test_kernel", config);

        // Act
        _timingInstrument.RecordKernelComplete("test_kernel", 10.0);
        _timingInstrument.RecordKernelComplete("test_kernel", 20.0);
        _timingInstrument.RecordKernelComplete("test_kernel", 30.0);

        var stats = _timingInstrument.GetTimingStatistics("test_kernel");

        // Assert
        Assert.AreEqual(3, stats.ExecutionCount);
        Assert.AreEqual(20.0, stats.AverageTimeMs, 0.01);
        Assert.AreEqual(10.0, stats.MinTimeMs, 0.01);
        Assert.AreEqual(30.0, stats.MaxTimeMs, 0.01);
        Assert.AreEqual(60.0, stats.TotalTimeMs, 0.01);
    }

    [Test]
    public void KernelTimingInstrument_UnknownKernel_ReturnsEmptyStats()
    {
        // Act
        var stats = _timingInstrument.GetTimingStatistics("unknown_kernel");

        // Assert
        Assert.AreEqual(0, stats.ExecutionCount);
        Assert.AreEqual(0, stats.TotalTimeMs);
        Assert.AreEqual("unknown_kernel", stats.KernelName);
    }

    [Test]
    public void KernelTimingInstrument_ComputesStandardDeviation()
    {
        // Arrange
        var config = CreateLaunchConfig();
        _timingInstrument.RecordKernelLaunch("test_kernel", config);

        // Act
        _timingInstrument.RecordKernelComplete("test_kernel", 10.0);
        _timingInstrument.RecordKernelComplete("test_kernel", 20.0);
        _timingInstrument.RecordKernelComplete("test_kernel", 30.0);

        var stats = _timingInstrument.GetTimingStatistics("test_kernel");

        // Assert
        // Standard deviation of [10, 20, 30] = 10
        Assert.AreEqual(10.0, stats.StdDevMs, 0.01);
    }

    [Test]
    public void ConsoleFusionDecisionLogger_LogDecision_OutputsCorrectFormat()
    {
        // Arrange
        var decision = new FusionDecision
        {
            OperationChain = "Add -> Mul",
            Fused = true,
            PatternType = FusionPatternType.ElementWise,
            RejectionReason = null,
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        // Act & Assert (should not throw)
        Assert.DoesNotThrow(() => _consoleLogger.LogDecision(decision));
    }

    [Test]
    public void ConsoleFusionDecisionLogger_LogRejectedDecision_OutputsReason()
    {
        // Arrange
        var decision = new FusionDecision
        {
            OperationChain = "Div -> Sub",
            Fused = false,
            PatternType = FusionPatternType.ElementWise,
            RejectionReason = "Shape mismatch",
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        // Act & Assert (should not throw)
        Assert.DoesNotThrow(() => _consoleLogger.LogDecision(decision));
    }

    [Test]
    public void ConsoleFusionDecisionLogger_LogTiming_OutputsCorrectFormat()
    {
        // Act & Assert (should not throw)
        Assert.DoesNotThrow(() => _consoleLogger.LogTiming("Test operation", 123.456));
    }

    [Test]
    public void PerformanceReportGenerator_GeneratesTextReport()
    {
        // Arrange
        var report = CreateSampleReport();

        // Act
        var text = _reportGenerator.GenerateTextReport(report);

        // Assert
        Assert.IsNotEmpty(text);
        Assert.IsTrue(text.Contains("Fusion Profiling Report"));
        Assert.IsTrue(text.Contains("Total Operations"));
    }

    [Test]
    public void PerformanceReportGenerator_GeneratesJsonReport()
    {
        // Arrange
        var report = CreateSampleReport();

        // Act
        var json = _reportGenerator.GenerateJsonReport(report);

        // Assert
        Assert.IsNotEmpty(json);
        // Verify it's valid JSON by deserializing
        var deserialized = System.Text.Json.JsonSerializer.Deserialize<FusionProfilingReport>(json);
        Assert.IsNotNull(deserialized);
    }

    [Test]
    public void PerformanceReportGenerator_GeneratesMarkdownReport()
    {
        // Arrange
        var report = CreateSampleReport();

        // Act
        var markdown = _reportGenerator.GenerateMarkdownReport(report);

        // Assert
        Assert.IsNotEmpty(markdown);
        Assert.IsTrue(markdown.Contains("# Fusion Profiling Report"));
        Assert.IsTrue(markdown.Contains("## Summary"));
        Assert.IsTrue(markdown.Contains("## Pattern Metrics"));
    }

    [Test]
    public void Profiler_ThreadSafe_MultipleConcurrentWrites()
    {
        // Arrange
        var tasks = new List<Task>();

        // Act
        for (int i = 0; i < 100; i++)
        {
            var index = i;
            tasks.Add(Task.Run(() =>
            {
                var decision = new FusionDecision
                {
                    OperationChain = $"Op{index} -> Op{index + 1}",
                    Fused = index % 2 == 0,
                    PatternType = index % 2 == 0 ? FusionPatternType.ElementWise : null,
                    RejectionReason = index % 2 != 0 ? "Test rejection" : null,
                    Timestamp = DateTime.UtcNow,
                    Metadata = new Dictionary<string, object>()
                };
                _profiler.RecordDecision(decision);
            }));
        }

        Task.WaitAll(tasks.ToArray());
        var report = _profiler.GetReport();

        // Assert
        Assert.AreEqual(100, report.Decisions.Count);
    }

    [Test]
    public void Profiler_PatternMetrics_UpdatesCorrectly()
    {
        // Arrange
        var decision1 = new FusionDecision
        {
            OperationChain = "Add -> Mul",
            Fused = true,
            PatternType = FusionPatternType.ElementWise,
            RejectionReason = null,
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        var decision2 = new FusionDecision
        {
            OperationChain = "Conv2D -> ReLU",
            Fused = true,
            PatternType = FusionPatternType.ConvActivation,
            RejectionReason = null,
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>()
        };

        // Act
        _profiler.RecordDecision(decision1);
        _profiler.RecordDecision(decision2);
        var report = _profiler.GetReport();

        // Assert
        Assert.IsTrue(report.PatternMetrics.ContainsKey("ElementWise"));
        Assert.IsTrue(report.PatternMetrics.ContainsKey("ConvActivation"));
        Assert.AreEqual(1, report.PatternMetrics["ElementWise"].Count);
        Assert.AreEqual(1, report.PatternMetrics["ConvActivation"].Count);
    }

    // Helper methods

    private FusionBackends.FusedOperation CreateTestFusedOperation()
    {
        return new FusionBackends.FusedOperation
        {
            Id = "test_fused_op",
            Type = "Fused_ElementWise",
            Name = "Fused_ElementWise_test",
            DataType = DataType.Float32,
            Layout = TensorLayout.Any,
            InputShape = new TensorShape { Dimensions = new[] { 10, 20 } },
            OutputShape = new TensorShape { Dimensions = new[] { 10, 20 } },
            Inputs = new[] { "input" },
            Outputs = new[] { "output" },
            Attributes = new Dictionary<string, object>(),
            ConstituentOperations = new List<Operation>(),
            Pattern = new FusionPatternDefinition
            {
                Name = "ElementWiseChain",
                OpTypeSequence = new[] { "Add", "Mul" },
                MatchStrategy = (ops) => true
            },
            IR = new FusionIR
            {
                Id = "ir_test",
                Nodes = new List<FusionOpNode>(),
                Variables = new List<FusionVariable>(),
                MemoryLayout = new MemoryLayout(),
                ComputeRequirements = new ComputeRequirements()
            },
            KernelSpec = new KernelSpecification
            {
                KernelName = "elementwise_kernel",
                ThreadBlockConfig = new ThreadBlockConfig { Total = 256, X = 256, Y = 1, Z = 1 },
                MemoryRequirements = new MemoryRequirements { SharedMemoryBytes = 0 },
                KernelType = KernelType.ElementWise
            }
        };
    }

    private KernelLaunchConfiguration CreateLaunchConfig()
    {
        return new KernelLaunchConfiguration
        {
            BlockDim = new ThreadBlockConfiguration { X = 256, Y = 1, Z = 1 },
            GridDim = new ThreadBlockConfiguration { X = 10, Y = 1, Z = 1 },
            SharedMemoryBytes = 1024,
            Parameters = new List<KernelLaunchParameter>()
        };
    }

    private FusionProfilingReport CreateSampleReport()
    {
        return new FusionProfilingReport
        {
            Decisions = new List<FusionDecision>
            {
                new FusionDecision
                {
                    OperationChain = "Add -> Mul -> ReLU",
                    Fused = true,
                    PatternType = FusionPatternType.ElementWise,
                    RejectionReason = null,
                    Timestamp = DateTime.UtcNow,
                    Metadata = new Dictionary<string, object>()
                }
            },
            KernelExecutions = new List<KernelExecutionRecord>
            {
                new KernelExecutionRecord
                {
                    KernelName = "elementwise_kernel",
                    DurationMs = 12.5,
                    ThreadCount = 256,
                    BlockCount = 10,
                    SharedMemoryBytes = 0,
                    Timestamp = DateTime.UtcNow
                }
            },
            Summary = new FusionSummary
            {
                TotalOperations = 3,
                FusedOperations = 3,
                FusedGroups = 1,
                FusionRate = 100.0,
                TotalKernelTimeMs = 12.5,
                AverageKernelTimeMs = 12.5,
                SuccessfulFusions = 1,
                FailedFusions = 0
            },
            PatternMetrics = new Dictionary<string, FusionPatternMetrics>
            {
                {
                    "ElementWise",
                    new FusionPatternMetrics
                    {
                        PatternName = "ElementWise",
                        Count = 1,
                        TotalTimeMs = 12.5,
                        AverageTimeMs = 12.5,
                        MinTimeMs = 12.5,
                        MaxTimeMs = 12.5,
                        EstimatedSpeedup = 2.0
                    }
                }
            }
        };
    }
}
