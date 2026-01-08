using Xunit;
using MLFramework.Compilation;
using MLFramework.Fusion;
using MLFramework.Fusion.Dynamic;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Tests.Compilation;

/// <summary>
/// Unit tests for LazyCompilationContext
/// </summary>
public class LazyCompilationContextTests
{
    [Fact]
    public void Create_WithValidParameters_CreatesContext()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var inputShapes = new List<int[]> { new[] { 32, 128 }, new[] { 32, 128 } };
        var outputShapes = new List<int[]> { new[] { 32, 128 } };

        // Act
        var context = LazyCompilationContext.Create(op, inputShapes, outputShapes);

        // Assert
        Assert.NotNull(context);
        Assert.Equal(op, context.Operation);
        Assert.Equal(inputShapes, context.InputShapes);
        Assert.Equal(outputShapes, context.OutputShapes);
        Assert.False(context.IsCompiled);
        Assert.Null(context.CompiledKernel);
        Assert.Equal(0, context.CompilationTimeMs);
    }

    [Fact]
    public void EnsureCompiled_FirstCall_CompilesKernel()
    {
        // Arrange
        var op = CreateTestOperation("Mul");
        var inputShapes = new List<int[]> { new[] { 64, 256 } };
        var outputShapes = new List<int[]> { new[] { 64, 256 } };
        var context = LazyCompilationContext.Create(op, inputShapes, outputShapes);
        var compiler = new MockCompiler();

        // Act
        context.EnsureCompiled(compiler);

        // Assert
        Assert.True(context.IsCompiled);
        Assert.NotNull(context.CompiledKernel);
        Assert.True(context.CompilationTimeMs >= 0);
        Assert.Equal(1, compiler.CompileCount);
    }

    [Fact]
    public void EnsureCompiled_MultipleCalls_CompilesOnlyOnce()
    {
        // Arrange
        var op = CreateTestOperation("Relu");
        var inputShapes = new List<int[]> { new[] { 32, 512 } };
        var outputShapes = new List<int[]> { new[] { 32, 512 } };
        var context = LazyCompilationContext.Create(op, inputShapes, outputShapes);
        var compiler = new MockCompiler();

        // Act
        context.EnsureCompiled(compiler);
        context.EnsureCompiled(compiler);
        context.EnsureCompiled(compiler);

        // Assert
        Assert.True(context.IsCompiled);
        Assert.Equal(1, compiler.CompileCount);
    }

    [Fact]
    public void Execute_BeforeCompilation_ThrowsInvalidOperationException()
    {
        // Arrange
        var op = CreateTestOperation("Sigmoid");
        var inputShapes = new List<int[]> { new[] { 32, 128 } };
        var outputShapes = new List<int[]> { new[] { 32, 128 } };
        var context = LazyCompilationContext.Create(op, inputShapes, outputShapes);
        var executor = new MockExecutor();
        var inputs = new List<Tensor>();
        var outputs = new List<Tensor>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => context.Execute(executor, inputs, outputs));
    }

    [Fact]
    public void Execute_AfterCompilation_ExecutesKernel()
    {
        // Arrange
        var op = CreateTestOperation("Tanh");
        var inputShapes = new List<int[]> { new[] { 16, 256 } };
        var outputShapes = new List<int[]> { new[] { 16, 256 } };
        var context = LazyCompilationContext.Create(op, inputShapes, outputShapes);
        var compiler = new MockCompiler();
        var executor = new MockExecutor();
        var inputs = new List<Tensor>();
        var outputs = new List<Tensor>();

        // Act
        context.EnsureCompiled(compiler);
        context.Execute(executor, inputs, outputs);

        // Assert
        Assert.Equal(1, executor.ExecuteCount);
    }

    [Fact]
    public void EnsureCompiled_ThreadSafe_DoesNotCompileMultipleTimes()
    {
        // Arrange
        var op = CreateTestOperation("Op");
        var inputShapes = new List<int[]> { new[] { 32, 128 } };
        var outputShapes = new List<int[]> { new[] { 32, 128 } };
        var context = LazyCompilationContext.Create(op, inputShapes, outputShapes);
        var compiler = new SlowMockCompiler(); // Simulates slow compilation

        // Act
        Parallel.For(0, 10, i => context.EnsureCompiled(compiler));

        // Assert
        Assert.True(context.IsCompiled);
        Assert.Equal(1, compiler.CompileCount);
    }

    [Fact]
    public void CompilationTime_IsRecordedCorrectly()
    {
        // Arrange
        var op = CreateTestOperation("Add");
        var inputShapes = new List<int[]> { new[] { 32, 128 } };
        var outputShapes = new List<int[]> { new[] { 32, 128 } };
        var context = LazyCompilationContext.Create(op, inputShapes, outputShapes);
        var compiler = new DelayedMockCompiler(50); // 50ms delay

        // Act
        context.EnsureCompiled(compiler);

        // Assert
        Assert.True(context.CompilationTimeMs >= 50);
        Assert.True(context.CompilationTimeMs < 200); // Should be close to 50ms
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

        public CompiledKernel Compile(Operation op, List<int[]> inputShapes, List<int[]> outputShapes)
        {
            CompileCount++;
            return new CompiledKernel
            {
                KernelId = "mock-kernel",
                SourceCode = "mock kernel",
                Binary = Array.Empty<byte>(),
                SpecializedShapes = inputShapes,
                IsGeneric = false,
                Signature = "mock-signature",
                EstimatedExecutionTimeNs = 1000
            };
        }

        public bool CanCompile(Operation op)
        {
            return true;
        }
    }

    private class SlowMockCompiler : IKernelCompiler
    {
        public int CompileCount { get; private set; }
        private readonly object _lock = new object();

        public CompiledKernel Compile(Operation op, List<int[]> inputShapes, List<int[]> outputShapes)
        {
            lock (_lock)
            {
                CompileCount++;
                Thread.Sleep(10); // Simulate compilation time
            }

            return new CompiledKernel
            {
                KernelId = "slow-kernel",
                SourceCode = "slow kernel",
                Binary = Array.Empty<byte>(),
                SpecializedShapes = inputShapes,
                IsGeneric = false,
                Signature = "slow-signature",
                EstimatedExecutionTimeNs = 1000
            };
        }

        public bool CanCompile(Operation op)
        {
            return true;
        }
    }

    private class DelayedMockCompiler : IKernelCompiler
    {
        private readonly int _delayMs;

        public int CompileCount { get; private set; }

        public DelayedMockCompiler(int delayMs)
        {
            _delayMs = delayMs;
        }

        public CompiledKernel Compile(Operation op, List<int[]> inputShapes, List<int[]> outputShapes)
        {
            CompileCount++;
            Thread.Sleep(_delayMs);

            return new CompiledKernel
            {
                KernelId = "delayed-kernel",
                SourceCode = "delayed kernel",
                Binary = Array.Empty<byte>(),
                SpecializedShapes = inputShapes,
                IsGeneric = false,
                Signature = "delayed-signature",
                EstimatedExecutionTimeNs = 1000
            };
        }

        public bool CanCompile(Operation op)
        {
            return true;
        }
    }

    private class MockExecutor : IKernelExecutor
    {
        public int ExecuteCount { get; private set; }

        public void Execute(CompiledKernel kernel, List<Tensor> inputs, List<Tensor> outputs)
        {
            ExecuteCount++;
        }
    }
}
