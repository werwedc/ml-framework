# Spec: CUDA Graph Unit Tests

## Overview
Implement a comprehensive test suite for CUDA Graph functionality. This spec defines unit and integration tests for all CUDA Graph components to ensure correctness and robustness.

## Requirements

### 1. Core Interfaces Tests
Test the core interfaces and base types.

```csharp
[TestClass]
public class CUDAGraphCoreInterfacesTests
{
    [TestMethod]
    public void CUDAGraphState_Enum_Values_AreCorrect()
    {
        Assert.AreEqual(6, Enum.GetValues(typeof(CUDAGraphState)).Length);
    }

    [TestMethod]
    public void CUDAGraphValidationResult_Valid_ReturnsTrue()
    {
        var result = new CUDAGraphValidationResult
        {
            IsValid = true,
            Errors = Array.Empty<string>(),
            Warnings = Array.Empty<string>(),
            OperationCount = 100
        };

        Assert.IsTrue(result.IsValid);
        Assert.AreEqual(0, result.Errors.Count);
        Assert.AreEqual(0, result.Warnings.Count);
        Assert.AreEqual(100, result.OperationCount);
    }

    [TestMethod]
    public void CUDAGraphValidationResult_Invalid_ReturnsFalse()
    {
        var result = new CUDAGraphValidationResult
        {
            IsValid = false,
            Errors = new[] { "Error 1", "Error 2" },
            Warnings = new[] { "Warning 1" },
            OperationCount = 0
        };

        Assert.IsFalse(result.IsValid);
        Assert.AreEqual(2, result.Errors.Count);
        Assert.AreEqual(1, result.Warnings.Count);
    }
}
```

### 2. Capture API Tests
Test the graph capture functionality.

```csharp
[TestClass]
public class CUDAGraphCaptureTests
{
    private CUDAStream _stream;
    private CUDAGraphCapture _capture;

    [TestInitialize]
    public void Setup()
    {
        _stream = new CUDAStream();
        _capture = new CUDAGraphCapture();
    }

    [TestCleanup]
    public void Cleanup()
    {
        _capture?.Dispose();
        _stream?.Dispose();
    }

    [TestMethod]
    public void BeginCapture_SetsIsCapturingToTrue()
    {
        _capture.BeginCapture(_stream);
        Assert.IsTrue(_capture.IsCapturing);
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidOperationException))]
    public void BeginCapture_WhenAlreadyCapturing_ThrowsException()
    {
        _capture.BeginCapture(_stream);
        _capture.BeginCapture(_stream);
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidOperationException))]
    public void EndCapture_WithoutBeginCapture_ThrowsException()
    {
        _capture.EndCapture();
    }

    [TestMethod]
    public void EndCapture_AfterBeginCapture_ReturnsGraph()
    {
        _capture.BeginCapture(_stream);
        var graph = _capture.EndCapture();

        Assert.IsNotNull(graph);
        Assert.IsFalse(_capture.IsCapturing);
    }

    [TestMethod]
    public void Dispose_WhileCapturing_AbortsCapture()
    {
        _capture.BeginCapture(_stream);
        _capture.Dispose();

        Assert.IsFalse(_capture.IsCapturing);
    }
}
```

### 3. Execution Engine Tests
Test the graph execution engine.

```csharp
[TestClass]
public class CUDAGraphExecutionTests
{
    private CUDAGraph _graph;
    private CUDAStream _stream;

    [TestInitialize]
    public void Setup()
    {
        _stream = new CUDAStream();
        // Note: Real graph capture requires GPU
        // For unit tests, we'll use mocks or skip
    }

    [TestCleanup]
    public void Cleanup()
    {
        _graph?.Dispose();
        _stream?.Dispose();
    }

    [TestMethod]
    public void CUDAGraph_AfterCreation_StateIsCreated()
    {
        // Create a mock graph for testing
        _graph = CreateMockGraph();

        Assert.AreEqual(CUDAGraphState.Created, _graph.State);
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void Execute_AfterDispose_ThrowsException()
    {
        _graph = CreateMockGraph();
        _graph.Dispose();
        _graph.Execute(_stream);
    }

    [TestMethod]
    public void Validate_ValidGraph_ReturnsValidResult()
    {
        _graph = CreateMockGraph();
        var result = _graph.Validate();

        Assert.IsNotNull(result);
        // Assert based on mock implementation
    }

    private CUDAGraph CreateMockGraph()
    {
        // Mock implementation for testing
        // In production, this would require actual CUDA graph capture
        return new CUDAGraph(IntPtr.Zero);
    }
}
```

### 4. Memory Pool Tests
Test the graph memory pool.

```csharp
[TestClass]
public class CUDAGraphMemoryPoolTests
{
    private CUDAGraphMemoryPool _pool;

    [TestInitialize]
    public void Setup()
    {
        _pool = new CUDAGraphMemoryPool(1024 * 1024); // 1MB
    }

    [TestCleanup]
    public void Cleanup()
    {
        _pool?.Dispose();
    }

    [TestMethod]
    public void Allocate_ValidSize_ReturnsBlock()
    {
        var block = _pool.Allocate(1024);

        Assert.IsNotNull(block);
        Assert.AreEqual(1024ul, block.Size);
    }

    [TestMethod]
    public void Allocate_MultipleTimes_IncreasesAllocatedBytes()
    {
        var size1 = _pool.AllocatedBytes;
        _pool.Allocate(1024);
        var size2 = _pool.AllocatedBytes;

        Assert.AreEqual(1024, size2 - size1);
    }

    [TestMethod]
    public void GetBlock_ValidId_ReturnsBlock()
    {
        var block = _pool.Allocate(1024);
        var retrieved = _pool.GetBlock(block.BlockId);

        Assert.AreEqual(block.BlockId, retrieved.BlockId);
    }

    [TestMethod]
    public void ReturnBlock_ValidId_SetsInUseToFalse()
    {
        var block = _pool.Allocate(1024);
        _pool.ReturnBlock(block.BlockId);
        var retrieved = _pool.GetBlock(block.BlockId);

        Assert.IsFalse(retrieved.InUse);
    }

    [TestMethod]
    public void Reset_ClearsAllBlocksToAvailable()
    {
        _pool.Allocate(1024);
        _pool.Allocate(2048);

        _pool.Reset();

        Assert.AreEqual(0, _pool.AllocatedBytes); // Or check blocks are available
    }

    [TestMethod]
    [ExpectedException(typeof(OutOfMemoryException))]
    public void Allocate_ExceedsMaxCapacity_ThrowsException()
    {
        var smallPool = new CUDAGraphMemoryPool(1000, 2000);
        smallPool.Allocate(3000);
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void Allocate_AfterDispose_ThrowsException()
    {
        _pool.Dispose();
        _pool.Allocate(1024);
    }

    [TestMethod]
    public void ConcurrentAllocations_ThreadSafe()
    {
        const int threadCount = 10;
        const int allocsPerThread = 5;
        var threads = new Thread[threadCount];
        var exceptions = new List<Exception>();

        for (int i = 0; i < threadCount; i++)
        {
            threads[i] = new Thread(() =>
            {
                try
                {
                    for (int j = 0; j < allocsPerThread; j++)
                    {
                        var block = _pool.Allocate(1024);
                        Thread.Sleep(1); // Simulate work
                        _pool.ReturnBlock(block.BlockId);
                    }
                }
                catch (Exception ex)
                {
                    lock (exceptions)
                    {
                        exceptions.Add(ex);
                    }
                }
            });
        }

        foreach (var thread in threads)
            thread.Start();

        foreach (var thread in threads)
            thread.Join();

        Assert.AreEqual(0, exceptions.Count);
    }
}
```

### 5. Graph Manager Tests
Test the graph manager functionality.

```csharp
[TestClass]
public class CUDAGraphManagerTests
{
    private CUDAGraphManager _manager;
    private CUDAStream _stream;

    [TestInitialize]
    public void Setup()
    {
        _manager = new CUDAGraphManager(captureIterations: 3);
        _stream = new CUDAStream();
    }

    [TestCleanup]
    public void Cleanup()
    {
        _manager?.Dispose();
        _stream?.Dispose();
    }

    [TestMethod]
    public void GetOrCaptureGraph_DuringWarmup_ReturnsNull()
    {
        var graph = _manager.GetOrCaptureGraph(
            "test",
            s => { /* Execute action */ },
            _stream);

        Assert.IsNull(graph);
        Assert.IsFalse(_manager.IsCaptureComplete);
    }

    [TestMethod]
    public void GetOrCaptureGraph_AfterWarmup_CapturesGraph()
    {
        // Run warmup iterations
        for (int i = 0; i < 3; i++)
        {
            _manager.GetOrCaptureGraph(
                "test",
                s => { /* Execute action */ },
                _stream);
        }

        var graph = _manager.GetOrCaptureGraph(
            "test",
            s => { /* Execute action */ },
            _stream);

        Assert.IsNotNull(graph);
        Assert.IsTrue(_manager.IsCaptureComplete);
    }

    [TestMethod]
    public void GetOrCaptureGraph_AlreadyCaptured_ReturnsCached()
    {
        // Warmup and capture
        for (int i = 0; i < 4; i++)
        {
            _manager.GetOrCaptureGraph(
                "test",
                s => { /* Execute action */ },
                _stream);
        }

        var graph1 = _manager.GetOrCaptureGraph(
            "test",
            s => { /* Execute action */ },
            _stream);
        var graph2 = _manager.GetOrCaptureGraph(
            "test",
            s => { /* Execute action */ },
            _stream);

        Assert.AreEqual(graph1.GraphId, graph2.GraphId);
    }

    [TestMethod]
    public void RemoveGraph_ValidId_RemovesGraph()
    {
        // Warmup and capture
        for (int i = 0; i < 4; i++)
        {
            _manager.GetOrCaptureGraph(
                "test",
                s => { /* Execute action */ },
                _stream);
        }

        var graph = _manager.GetGraph("test");
        Assert.IsNotNull(graph);

        _manager.RemoveGraph("test");
        graph = _manager.GetGraph("test");
        Assert.IsNull(graph);
    }

    [TestMethod]
    public void ClearGraphs_RemovesAllGraphs()
    {
        // Warmup and capture multiple graphs
        for (int i = 0; i < 5; i++)
        {
            _manager.GetOrCaptureGraph(
                $"graph{i}",
                s => { /* Execute action */ },
                _stream);
        }

        Assert.IsTrue(_manager.GraphCount > 0);
        _manager.ClearGraphs();
        Assert.AreEqual(0, _manager.GraphCount);
    }
}
```

### 6. Fallback Mechanism Tests
Test the fallback mechanism.

```csharp
[TestClass]
public class CUDAGraphFallbackTests
{
    [TestMethod]
    public void FallbackHandler_NeverCapture_AlwaysUsesFallback()
    {
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.NeverCapture);

        Assert.IsTrue(handler.ShouldUseFallback);
    }

    [TestMethod]
    public void FallbackHandler_CaptureOrFallback_UsesFallbackOnFailure()
    {
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        var stream = new CUDAStream();

        handler.ExecuteWithFallback(
            () => throw new Exception("Capture failed"),
            s => { /* Regular execution */ },
            stream);

        Assert.IsTrue(handler.ShouldUseFallback);
    }

    [TestMethod]
    public void FallbackHandler_RetryThenFallback_RetriesMaxAttempts()
    {
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.RetryThenFallback, 3);
        var stream = new CUDAStream();
        var attempts = 0;

        for (int i = 0; i < 5; i++)
        {
            handler.ExecuteWithFallback(
                () =>
                {
                    attempts++;
                    throw new Exception("Capture failed");
                },
                s => { /* Regular execution */ },
                stream);
        }

        Assert.AreEqual(3, handler.CaptureAttempts);
    }

    [TestMethod]
    public void FallbackHandler_Reset_ClearsState()
    {
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        var stream = new CUDAStream();

        handler.ExecuteWithFallback(
            () => throw new Exception("Capture failed"),
            s => { /* Regular execution */ },
            stream);

        Assert.IsTrue(handler.ShouldUseFallback);
        Assert.IsTrue(handler.CaptureAttempts > 0);

        handler.Reset();

        Assert.IsFalse(handler.ShouldUseFallback);
        Assert.AreEqual(0, handler.CaptureAttempts);
    }
}
```

## Implementation Details

### File Structure
- **File**: `tests/CUDA/Graphs/CUDAGraphCoreInterfacesTests.cs`
- **File**: `tests/CUDA/Graphs/CUDAGraphCaptureTests.cs`
- **File**: `tests/CUDA/Graphs/CUDAGraphExecutionTests.cs`
- **File**: `tests/CUDA/Memory/CUDAGraphMemoryPoolTests.cs`
- **File**: `tests/CUDA/Graphs/CUDAGraphManagerTests.cs`
- **File**: `tests/CUDA/Graphs/CUDAGraphFallbackTests.cs`

### Dependencies
- All CUDA Graph components
- MSTest or NUnit for test framework
- Moq or NSubstitute for mocking (if needed)
- System.Threading for concurrent tests

### Test Organization
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **Performance Tests**: Measure and validate performance improvements

### Test Categories
- **Fast**: Tests that don't require GPU
- **GPU**: Tests that require GPU hardware
- **Slow**: Tests that take significant time

## Success Criteria
- All unit tests pass
- Tests cover all public APIs
- Tests cover edge cases
- Tests cover error conditions
- Tests are fast (unit tests)
- Integration tests work correctly
- Performance tests show expected improvements

## Testing Requirements

### Test Coverage Goals
- **Core Interfaces**: 100% coverage
- **Capture API**: 90%+ coverage
- **Execution Engine**: 85%+ coverage
- **Memory Pool**: 90%+ coverage
- **Graph Manager**: 85%+ coverage
- **Fallback Mechanism**: 90%+ coverage

### Integration Tests
- Test end-to-end graph capture and execution
- Test with actual model training (requires GPU)
- Test performance improvements
- Test with multiple graphs
- Test with weight updates
