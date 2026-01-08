using Xunit;
using MLFramework.Compilation;
using MLFramework.Fusion;
using MLFramework.Fusion.Dynamic;
using MLFramework.Shapes;

namespace MLFramework.Tests.Compilation;

/// <summary>
/// Unit tests for LazyCompilationManager
/// </summary>
public class LazyCompilationManagerTests
{
    private readonly MockCompiler _compiler;
    private readonly LRUKernelCache<CompiledKernel> _cache;
    private readonly LazyCompilationManager _manager;

    public LazyCompilationManagerTests()
    {
        _compiler = new MockCompiler();
        _cache = new LRUKernelCache<CompiledKernel>(100);
        _manager = new LazyCompilationManager(_cache, _compiler);
    }

    [Fact]
    public void GetOrCompile_FirstCall_CompilesKernel()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var shapes = new List<int[]> { new[] { 32, 128 }, new[] { 32, 128 } };

        // Act
        var kernel = _manager.GetOrCompile(op, shapes);

        // Assert
        Assert.NotNull(kernel);
        Assert.Equal(1, _compiler.CompileCount);
        Assert.Equal(1, _manager.GetCompilationStats().TotalCompilations);
    }

    [Fact]
    public void GetOrCompile_SecondCallWithSameShapes_UsesCache()
    {
        // Arrange
        var op = CreateTestOperation("Mul");
        var shapes = new List<int[]> { new[] { 64, 256 } };

        // Act
        var kernel1 = _manager.GetOrCompile(op, shapes);
        var kernel2 = _manager.GetOrCompile(op, shapes);

        // Assert
        Assert.Same(kernel1, kernel2);
        Assert.Equal(1, _compiler.CompileCount);
        Assert.Equal(1, _manager.GetCompilationStats().TotalCompilations);
        Assert.Equal(1, _manager.GetCompilationStats().CacheHits);
    }

    [Fact]
    public void GetOrCompile_DifferentShapes_CompilesNewKernel()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var shapes1 = new List<int[]> { new[] { 32, 128 } };
        var shapes2 = new List<int[]> { new[] { 64, 256 } };

        // Act
        _manager.GetOrCompile(op, shapes1);
        _manager.GetOrCompile(op, shapes2);

        // Assert
        Assert.Equal(2, _compiler.CompileCount);
        Assert.Equal(2, _manager.GetCompilationStats().TotalCompilations);
    }

    [Fact]
    public void GetOrCompile_StatsTracking_TracksCorrectly()
    {
        // Arrange
        var op = CreateTestOperation("Relu");
        var shapes = new List<int[]> { new[] { 32, 512 } };

        // Act
        _manager.GetOrCompile(op, shapes);
        _manager.GetOrCompile(op, shapes);
        _manager.GetOrCompile(op, shapes);

        // Assert
        var stats = _manager.GetCompilationStats();
        Assert.Equal(1, stats.TotalCompilations);
        Assert.Equal(2, stats.CacheHits);
        Assert.Equal(1, stats.CacheMisses);
        Assert.Equal(1, stats.UniqueKernels);
        Assert.Equal(1.0, stats.HitRate);
    }

    [Fact]
    public void Precompile_WithMultipleShapeVariants_CompilesAll()
    {
        // Arrange
        var op = CreateTestOperation("Conv2D");
        var shapeVariants = new List<List<int[]>>
        {
            new() { new[] { 1, 3, 224, 224 } },
            new() { new[] { 1, 3, 112, 112 } },
            new() { new[] { 1, 3, 56, 56 } }
        };

        // Act
        _manager.Precompile(op, shapeVariants);

        // Assert
        Assert.Equal(3, _compiler.CompileCount);
        Assert.Equal(3, _manager.GetCompilationStats().TotalCompilations);
    }

    [Fact]
    public void ClearCache_AfterCompilation_emptiesCache()
    {
        // Arrange
        var op = CreateTestOperation("Sigmoid");
        var shapes = new List<int[]> { new[] { 32, 128 } };
        _manager.GetOrCompile(op, shapes);

        // Act
        _manager.ClearCache();

        // Assert
        Assert.Equal(0, _manager.Cache.CurrentSize);
        Assert.Equal(0, _manager.GetCompilationStats().UniqueKernels);
    }

    [Fact]
    public void GetCompilationStats_ReturnsCorrectStats()
    {
        // Arrange
        var op = CreateTestOperation("Tanh");
        _manager.GetOrCompile(op, new List<int[]> { new[] { 16, 256 } });
        _manager.GetOrCompile(op, new List<int[]> { new[] { 16, 256 } });
        _manager.GetOrCompile(op, new List<int[]> { new[] { 32, 512 } });

        // Act
        var stats = _manager.GetCompilationStats();

        // Assert
        Assert.Equal(2, stats.TotalCompilations);
        Assert.Equal(1, stats.CacheHits);
        Assert.Equal(2, stats.CacheMisses);
        Assert.Equal(2, stats.UniqueKernels);
        Assert.Equal(0.5, stats.HitRate);
    }

    [Fact]
    public void GetCompilationStats_ToReport_GeneratesReadableReport()
    {
        // Arrange
        var op = CreateTestOperation("Op");
        _manager.GetOrCompile(op, new List<int[]> { new[] { 32, 128 } });

        // Act
        var stats = _manager.GetCompilationStats();
        var report = stats.ToReport();

        // Assert
        Assert.Contains("Compilation Statistics", report);
        Assert.Contains("Total Compilations: 1", report);
        Assert.Contains("Cache Hits: 0", report);
    }

    [Fact]
    public void CreateContext_CreatesContextWithShapes()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var inputShapes = new List<int[]> { new[] { 32, 128 }, new[] { 32, 128 } };

        // Act
        var context = _manager.CreateContext(op, inputShapes);

        // Assert
        Assert.NotNull(context);
        Assert.False(context.IsCompiled);
        Assert.Null(context.CompiledKernel);
        Assert.Equal(inputShapes, context.InputShapes);
    }

    [Fact]
    public void GetOrCompile_TracksCompilationTime()
    {
        // Arrange
        var op = CreateTestOperation("Op");
        var shapes = new List<int[]> { new[] { 32, 128 } };

        // Act
        _manager.GetOrCompile(op, shapes);

        // Assert
        var stats = _manager.GetCompilationStats();
        Assert.True(stats.TotalCompilationTimeMs >= 0);
        Assert.True(stats.AverageCompilationTimeMs >= 0);
    }

    private Operation CreateTestOperation(string type)
    {
        return new TestOperation
        {
            Id = $"{type.ToLower()}_1",
            Type = type,
            Name = $"Test {type}",
            DataType = DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = new TensorShape(new[] { 32, 128 }),
            OutputShape = new TensorShape(new[] { 32, 128 }),
            Inputs = Array.Empty<string>(),
            Outputs = new[] { "output" },
            Attributes = new Dictionary<string, object>()
        };
    }

    private record TestOperation : Operation;

    private class MockCompiler : IKernelCompiler
    {
        public int CompileCount { get; private set; }
        private int _kernelIdCounter = 0;

        public CompiledKernel Compile(Operation op, List<int[]> inputShapes, List<int[]> outputShapes)
        {
            CompileCount++;
            return new CompiledKernel
            {
                KernelId = $"kernel-{_kernelIdCounter++}",
                SourceCode = $"kernel for {op.Type}",
                Binary = Array.Empty<byte>(),
                SpecializedShapes = inputShapes,
                IsGeneric = false,
                Signature = $"{op.Type}({string.Join(",", inputShapes.SelectMany(s => s))})",
                EstimatedExecutionTimeNs = 1000
            };
        }

        public bool CanCompile(Operation op)
        {
            return true; // Mock compiler can compile any operation
        }
    }
}
