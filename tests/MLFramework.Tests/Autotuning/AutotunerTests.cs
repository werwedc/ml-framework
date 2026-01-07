using NUnit.Framework;
using MLFramework.Autotuning;
using MLFramework.Core;
using MLFramework.Fusion;
using Backends = MLFramework.Fusion.Backends;
using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Tests.Autotuning;

/// <summary>
/// Unit tests for the autotuning system
/// </summary>
[TestFixture]
public class AutotunerTests
{
    private MockKernelExecutor _mockExecutor = null!;
    private MockPerformanceProfiler _mockProfiler = null!;
    private MockTensorGenerator _mockTensorGenerator = null!;
    private MockDeviceQuery _mockDeviceQuery = null!;
    private MockLogger _mockLogger = null!;
    private InMemoryTuningCache _cache = null!;
    private SearchSpaceGenerator _searchSpaceGenerator = null!;
    private BenchmarkRunner _benchmarkRunner = null!;
    private Autotuner _autotuner = null!;

    [SetUp]
    public void Setup()
    {
        _mockExecutor = new MockKernelExecutor();
        _mockProfiler = new MockPerformanceProfiler();
        _mockTensorGenerator = new MockTensorGenerator();
        _mockDeviceQuery = new MockDeviceQuery();
        _mockLogger = new MockLogger();
        _cache = new InMemoryTuningCache();
        _searchSpaceGenerator = new SearchSpaceGenerator();
        _benchmarkRunner = new BenchmarkRunner(_mockExecutor, _mockProfiler);

        _autotuner = new Autotuner(
            _cache,
            _searchSpaceGenerator,
            _benchmarkRunner,
            _mockTensorGenerator,
            _mockDeviceQuery,
            _mockLogger);
    }

    [Test]
    public void Tune_SelectsBestConfiguration()
    {
        // Arrange
        var fusedOp = CreateSimpleFusedOperation();
        var options = new AutotuningOptions { MaxIterations = 5 };

        // Act
        var result = _autotuner.Tune(fusedOp, options);

        // Assert
        Assert.IsNotNull(result.BestConfiguration);
        Assert.Less(result.BestExecutionTimeMs, result.Benchmarks.First().ExecutionTimeMs + 0.01);
        Assert.AreEqual(result.Benchmarks.Min(b => b.ExecutionTimeMs), result.BestExecutionTimeMs);
    }

    [Test]
    public void Cache_StoresAndRetrievesResult()
    {
        // Arrange
        var fusedOp = CreateSimpleFusedOperation();
        var device = _mockDeviceQuery.GetCurrentDeviceInfo();
        var result = CreateAutotuningResult();

        // Act
        _cache.Put(fusedOp, result, device);
        var retrieved = _cache.Get(fusedOp, device);

        // Assert
        Assert.IsNotNull(retrieved);
        Assert.AreEqual(result.BestConfiguration, retrieved.Result.BestConfiguration);
    }

    [Test]
    public void SearchSpaceGenerator_GeneratesValidConfigs()
    {
        // Arrange
        var fusedOp = CreateSimpleFusedOperation();
        var device = _mockDeviceQuery.GetCurrentDeviceInfo();

        // Act
        var configs = _searchSpaceGenerator.GenerateSearchSpace(
            fusedOp,
            device,
            SearchStrategy.GridSearch,
            10);

        // Assert
        Assert.IsNotEmpty(configs);
        Assert.True(configs.All(c => c.BlockDim.Total() <= device.MaxThreadsPerBlock));
    }

    [Test]
    public void BenchmarkRunner_MeasuresExecutionTime()
    {
        // Arrange
        var fusedOp = CreateSimpleFusedOperation();
        var config = CreateLaunchConfiguration();
        var input = _mockTensorGenerator.GenerateRandomTensor(fusedOp.InputShape, fusedOp.DataType);
        var options = new AutotuningOptions { WarmupRuns = 2, MeasurementRuns = 3 };

        // Act
        var result = _benchmarkRunner.Benchmark(fusedOp, config, input, options);

        // Assert
        Assert.Greater(result.ExecutionTimeMs, 0);
    }

    [Test]
    public void Autotuner_UsesCacheWhenAvailable()
    {
        // Arrange
        var fusedOp = CreateSimpleFusedOperation();
        var device = _mockDeviceQuery.GetCurrentDeviceInfo();
        var options = new AutotuningOptions { MaxIterations = 5 };

        // First tune to populate cache
        _autotuner.Tune(fusedOp, options);

        // Act - second tune should hit cache
        var result = _autotuner.Tune(fusedOp, options);

        // Assert
        Assert.IsTrue(result.CacheHit);
        Assert.Greater(_cache.Count, 0);
    }

    [Test]
    public void TuningCacheEntry_HitCountIncrements()
    {
        // Arrange
        var fusedOp = CreateSimpleFusedOperation();
        var device = _mockDeviceQuery.GetCurrentDeviceInfo();
        var result = CreateAutotuningResult();

        // Act
        _cache.Put(fusedOp, result, device);
        var firstRetrieve = _cache.Get(fusedOp, device);
        var secondRetrieve = _cache.Get(fusedOp, device);
        var thirdRetrieve = _cache.Get(fusedOp, device);

        // Assert
        Assert.IsNotNull(thirdRetrieve);
        Assert.AreEqual(2, thirdRetrieve.HitCount); // Retrieved twice after initial put
    }

    // Helper methods
    private MLFramework.Fusion.FusedOperation CreateSimpleFusedOperation()
    {
        var pattern = new FusionPatternDefinition
        {
            Name = "TestPattern",
            Description = "Test pattern",
            PatternType = Fusion.PatternType.ElementWise,
            MatchingFunction = _ => false
        };

        var ir = new FusionIR
        {
            Id = "test_ir",
            Nodes = Array.Empty<FusionOpNode>(),
            Variables = Array.Empty<FusionVariable>(),
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = TensorLayout.RowMajor,
                SharedMemoryBytes = 1024,
                RegisterBytes = 256
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 10,
                ThreadsPerBlock = 128,
                RequiresSharedMemory = false,
                RequiresAtomicOps = false
            }
        };

        var kernelSpec = new KernelSpecification
        {
            KernelName = "test_kernel",
            Strategy = FusionStrategy.ElementWise,
            InputTensors = Array.Empty<FusionVariable>(),
            OutputTensors = Array.Empty<FusionVariable>(),
            TemporaryMemoryBytes = 0,
            RegisterBytes = 256,
            ThreadBlockConfig = new Backends.ThreadBlockConfiguration(),
            CompilationFlags = Array.Empty<string>(),
            Parameters = Array.Empty<Backends.KernelLaunchParameter>()
        };

        return MLFramework.Fusion.FusedOperation.Create(
            "test_fused_op",
            Array.Empty<Operation>(),
            pattern,
            ir,
            kernelSpec);
    }

    private AutotuningResult CreateAutotuningResult()
    {
        return new AutotuningResult
        {
            BestConfiguration = CreateLaunchConfiguration(),
            BestExecutionTimeMs = 10.0,
            Benchmarks = new List<TuningBenchmarkResult>
            {
                new TuningBenchmarkResult
                {
                    Configuration = CreateLaunchConfiguration(),
                    ExecutionTimeMs = 10.0,
                    MemoryBandwidthGBps = 100.0,
                    SMUtilizationPercent = 50
                }
            },
            CacheHit = false,
            Timestamp = DateTime.UtcNow
        };
    }

    private Backends.KernelLaunchConfiguration CreateLaunchConfiguration()
    {
        return new Backends.KernelLaunchConfiguration
        {
            BlockDim = new Backends.ThreadBlockConfiguration { X = 128, Y = 1, Z = 1 },
            GridDim = new Backends.ThreadBlockConfiguration { X = 10, Y = 1, Z = 1 },
            SharedMemoryBytes = 1024,
            Parameters = Array.Empty<Backends.KernelLaunchParameter>()
        };
    }

    // Mock implementations
    private class MockKernelExecutor : IKernelExecutor
    {
        private int _executionCount = 0;

        public int ExecutionCount => _executionCount;

        public void ExecuteFusedKernel(MLFramework.Fusion.FusedOperation fusedOp, Backends.KernelLaunchConfiguration config, Tensor input)
        {
            _executionCount++;
            // Simulate execution with varying times for benchmarking
        }

        public void Synchronize()
        {
            // Simulate synchronization
        }
    }

    private class MockPerformanceProfiler : IPerformanceProfiler
    {
        private int _measureCount = 0;

        public int MeasureCount => _measureCount;

        public double MeasureExecution(Action action)
        {
            _measureCount++;
            action();
            // Return varying execution times
            return 1.0 + (_measureCount * 0.1); // 1.0, 1.1, 1.2, etc.
        }

        public double MeasureMemoryBandwidth(MLFramework.Fusion.FusedOperation fusedOp, Backends.KernelLaunchConfiguration config)
        {
            return 100.0; // Fixed value
        }

        public int MeasureSMUtilization(MLFramework.Fusion.FusedOperation fusedOp, Backends.KernelLaunchConfiguration config)
        {
            return 50; // Fixed value
        }
    }

    private class MockTensorGenerator : ITensorGenerator
    {
        public Tensor GenerateRandomTensor(TensorShape shape, MLFramework.Core.DataType dataType)
        {
            // Return a mock tensor
            return new Tensor(shape.Dimensions.ToArray(), RitterFramework.Core.DataType.Float32);
        }
    }

    private class MockDeviceQuery : IDeviceQuery
    {
        public DeviceInfo GetCurrentDeviceInfo()
        {
            return new DeviceInfo
            {
                DeviceName = "MockDevice",
                ComputeCapability = 75,
                SMCount = 80,
                TotalMemoryBytes = 16L * 1024 * 1024 * 1024, // 16 GB
                MaxThreadsPerBlock = 1024,
                MaxSharedMemoryPerBlock = 48 * 1024, // 48 KB
                Architecture = "Ampere"
            };
        }
    }

    private class MockLogger : MLFramework.Autotuning.ILogger
    {
        public List<string> LoggedMessages = new();

        public void LogInformation(string message, params object[] args)
        {
            LoggedMessages.Add(string.Format(message, args));
        }

        public void LogWarning(string message, params object[] args)
        {
            LoggedMessages.Add($"WARNING: {string.Format(message, args)}");
        }

        public void LogError(string message, params object[] args)
        {
            LoggedMessages.Add($"ERROR: {string.Format(message, args)}");
        }
    }
}
