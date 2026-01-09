# Spec: Mobile Runtime Tests

## Overview
Implement comprehensive unit and integration tests for the mobile runtime system.

## Requirements
- Unit tests for all components
- Integration tests for end-to-end workflows
- Cross-platform tests
- Performance regression tests
- Memory leak detection
- Accuracy validation

## Test Structure

### 1. Tensor Operations Tests
```csharp
[TestClass]
public class TensorOperationsTests
{
    [TestMethod]
    public void Add_TwoTensors_ReturnsCorrectResult()
    {
        var a = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var b = Tensor.FromArray(new[] { 5f, 6f, 7f, 8f }, new[] { 4 });

        var result = TensorOperations.Add(a, b);

        CollectionAssert.AreEqual(
            new[] { 6f, 8f, 10f, 12f },
            result.ToArray<float>()
        );
    }

    [TestMethod]
    public void Relu_PositiveValues_ReturnsUnchanged()
    {
        var input = Tensor.FromArray(new[] { 1f, 2f, 3f }, new[] { 3 });

        var result = TensorOperations.Relu(input);

        CollectionAssert.AreEqual(
            new[] { 1f, 2f, 3f },
            result.ToArray<float>()
        );
    }

    [TestMethod]
    public void Relu_NegativeValues_ReturnsZero()
    {
        var input = Tensor.FromArray(new[] { -1f, -2f, -3f }, new[] { 3 });

        var result = TensorOperations.Relu(input);

        CollectionAssert.AreEqual(
            new[] { 0f, 0f, 0f },
            result.ToArray<float>()
        );
    }

    [TestMethod]
    public void MatMul_2x2Matrices_ReturnsCorrectResult()
    {
        var a = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var b = Tensor.FromArray(new[] { 5f, 6f, 7f, 8f }, new[] { 2, 2 });

        var result = TensorOperations.MatMul(a, b);

        CollectionAssert.AreEqual(
            new[] { 19f, 22f, 43f, 50f },
            result.ToArray<float>()
        );
    }
}
```

### 2. Memory Pool Tests
```csharp
[TestClass]
public class MemoryPoolTests
{
    private DefaultMemoryPool _pool;

    [TestInitialize]
    public void Setup()
    {
        _pool = new DefaultMemoryPool(16 * 1024 * 1024);
    }

    [TestCleanup]
    public void Cleanup()
    {
        _pool.Dispose();
    }

    [TestMethod]
    public void Allocate_WithinLimit_Succeeds()
    {
        var ptr = _pool.Allocate(1024, DataType.Float32);

        Assert.AreNotEqual(IntPtr.Zero, ptr);

        _pool.Free(ptr, 1024);
    }

    [TestMethod]
    public void Allocate_ExceedsLimit_ThrowsException()
    {
        _pool.SetMemoryLimit(1024);

        Assert.ThrowsException<OutOfMemoryException>(() =>
        {
            _pool.Allocate(2048, DataType.Float32);
        });
    }

    [TestMethod]
    public void FreeAndReallocate_ReusesBlock()
    {
        var ptr1 = _pool.Allocate(1024, DataType.Float32);
        _pool.Free(ptr1, 1024);
        var ptr2 = _pool.Allocate(1024, DataType.Float32);

        Assert.AreEqual(ptr1, ptr2);

        _pool.Free(ptr2, 1024);
    }

    [TestMethod]
    public void MultipleAllocations_MemoryPoolEfficient()
    {
        const int count = 1000;
        var ptrs = new IntPtr[count];

        for (int i = 0; i < count; i++)
        {
            ptrs[i] = _pool.Allocate(1024, DataType.Float32);
        }

        var stats = _pool.GetStats();
        Assert.IsTrue(stats.CacheHits > 0);

        foreach (var ptr in ptrs)
        {
            _pool.Free(ptr, 1024);
        }
    }
}
```

### 3. Model Format Tests
```csharp
[TestClass]
public class ModelFormatTests
{
    [TestMethod]
    public void WriteAndReadModel_RoundTrip_Success()
    {
        var config = new ModelWriterConfig
        {
            ModelName = "TestModel",
            Inputs = new[] { ModelSerializer.CreateSpec("input", DataType.Float32, 1, 28, 28) },
            Outputs = new[] { ModelSerializer.CreateSpec("output", DataType.Float32, 10) },
            ConstantTensors = new ConstantTensor[0],
            Operators = new OperatorDescriptor[0]
        };

        var data = ModelSerializer.Save(config);
        var model = ModelSerializer.Load(data);

        Assert.AreEqual("TestModel", model.Metadata.Name);
        Assert.AreEqual(1, model.Inputs.Length);
        Assert.AreEqual("input", model.Inputs[0].Name);
        Assert.AreEqual(DataType.Float32, model.Inputs[0].DataType);
    }

    [TestMethod]
    public void LoadModel_InvalidChecksum_ThrowsException()
    {
        var config = new ModelWriterConfig
        {
            ModelName = "TestModel",
            Inputs = new InputOutputSpec[0],
            Outputs = new InputOutputSpec[0],
            ConstantTensors = new ConstantTensor[0],
            Operators = new OperatorDescriptor[0]
        };

        var data = ModelSerializer.Save(config);
        // Corrupt last 4 bytes (checksum)
        data[data.Length - 1] ^= 0xFF;

        Assert.ThrowsException<InvalidDataException>(() =>
        {
            ModelSerializer.Load(data);
        });
    }
}
```

### 4. CPU Backend Tests
```csharp
[TestClass]
public class CpuBackendTests
{
    private CpuBackend _backend;
    private ITensorFactory _tensorFactory;
    private DefaultMemoryPool _pool;

    [TestInitialize]
    public void Setup()
    {
        _pool = new DefaultMemoryPool();
        _tensorFactory = new TensorFactory(_pool);
        _backend = new CpuBackend(_pool, _tensorFactory);
    }

    [TestCleanup]
    public void Cleanup()
    {
        _backend.Dispose();
        _pool.Dispose();
    }

    [TestMethod]
    public void ExecuteRelu_NegativeInput_ReturnsZero()
    {
        var input = _tensorFactory.CreateTensor(new[] { -1f, -2f, -3f }, new[] { 3 });
        var op = new OperatorDescriptor
        {
            Type = OperatorType.Relu,
            InputTensorIds = new uint[] { 1 },
            OutputTensorIds = new uint[] { 2 },
            Parameters = new Dictionary<string, object>()
        };

        var result = _backend.Execute(op, new[] { input }, op.Parameters);

        CollectionAssert.AreEqual(
            new[] { 0f, 0f, 0f },
            result.ToArray<float>()
        );
    }

    [TestMethod]
    public void ExecuteAdd_TwoTensors_ReturnsCorrectResult()
    {
        var a = _tensorFactory.CreateTensor(new[] { 1f, 2f, 3f }, new[] { 3 });
        var b = _tensorFactory.CreateTensor(new[] { 4f, 5f, 6f }, new[] { 3 });
        var op = new OperatorDescriptor
        {
            Type = OperatorType.Add,
            InputTensorIds = new uint[] { 1, 2 },
            OutputTensorIds = new uint[] { 3 },
            Parameters = new Dictionary<string, object>()
        };

        var result = _backend.Execute(op, new[] { a, b }, op.Parameters);

        CollectionAssert.AreEqual(
            new[] { 5f, 7f, 9f },
            result.ToArray<float>()
        );
    }

    [TestMethod]
    public void ExecuteBatch_MultipleOperators_ReturnsCorrectResult()
    {
        var input = _tensorFactory.CreateTensor(new[] { -1f, -2f, -3f }, new[] { 3 });

        var ops = new[]
        {
            new OperatorDescriptor
            {
                Type = OperatorType.Relu,
                InputTensorIds = new uint[] { 1 },
                OutputTensorIds = new uint[] { 2 },
                Parameters = new Dictionary<string, object>()
            },
            new OperatorDescriptor
            {
                Type = OperatorType.Add,
                InputTensorIds = new uint[] { 2, 2 }, // Add to itself (multiply by 2)
                OutputTensorIds = new uint[] { 3 },
                Parameters = new Dictionary<string, object>()
            }
        };

        var tensorRegistry = new Dictionary<uint, ITensor>
        {
            { 1, input }
        };

        var results = _backend.ExecuteBatch(ops, tensorRegistry);

        Assert.AreEqual(3u, results.Last().Id);
        CollectionAssert.AreEqual(
            new[] { 0f, 0f, 0f },
            tensorRegistry[3].ToArray<float>()
        );
    }
}
```

### 5. Model Loader Tests
```csharp
[TestClass]
public class MobileModelTests
{
    private RuntimeMobileRuntime _runtime;

    [TestInitialize]
    public void Setup()
    {
        _runtime = new RuntimeMobileRuntime();
    }

    [TestCleanup]
    public void Cleanup()
    {
        // Clean up test models
    }

    [TestMethod]
    public void LoadModel_ValidFile_Succeeds()
    {
        // Create test model file
        var config = CreateTestModelConfig();
        var modelData = ModelSerializer.Save(config);
        var filePath = Path.GetTempFileName();
        File.WriteAllBytes(filePath, modelData);

        var model = _runtime.LoadModel(filePath);

        Assert.IsNotNull(model);
        Assert.AreEqual("TestModel", model.Name);
        Assert.AreEqual(1, model.Inputs.Length);

        File.Delete(filePath);
    }

    [TestMethod]
    public void Predict_ValidInputs_ReturnsExpectedOutputs()
    {
        var config = CreateTestModelConfig();
        var modelData = ModelSerializer.Save(config);
        var model = _runtime.LoadModel(modelData);

        var input = _tensorFactory.CreateTensor(new[] { 1f, 2f, 3f }, new[] { 3 });
        var outputs = model.Predict(new[] { input });

        Assert.AreEqual(1, outputs.Length);
        Assert.AreEqual(DataType.Float32, outputs[0].DataType);
    }

    [TestMethod]
    public async Task PredictAsync_ValidInputs_ReturnsExpectedOutputs()
    {
        var config = CreateTestModelConfig();
        var modelData = ModelSerializer.Save(config);
        var model = _runtime.LoadModel(modelData);

        var input = _tensorFactory.CreateTensor(new[] { 1f, 2f, 3f }, new[] { 3 });
        var outputs = await model.PredictAsync(new[] { input });

        Assert.AreEqual(1, outputs.Length);
        Assert.AreEqual(DataType.Float32, outputs[0].DataType);
    }

    private ModelWriterConfig CreateTestModelConfig()
    {
        // Create minimal test model
        return new ModelWriterConfig
        {
            ModelName = "TestModel",
            Inputs = new[] { ModelSerializer.CreateSpec("input", DataType.Float32, 3) },
            Outputs = new[] { ModelSerializer.CreateSpec("output", DataType.Float32, 3) },
            ConstantTensors = new ConstantTensor[0],
            Operators = new[]
            {
                new OperatorDescriptor
                {
                    Type = OperatorType.Relu,
                    InputTensorIds = new uint[] { 1 },
                    OutputTensorIds = new uint[] { 2 },
                    Parameters = new Dictionary<string, object>()
                }
            }
        };
    }
}
```

### 6. Integration Tests
```csharp
[TestClass]
public class MobileRuntimeIntegrationTests
{
    [TestMethod]
    public void EndToEnd_Inference_Succeeds()
    {
        // 1. Create and save model
        var config = CreateTestModelConfig();
        var modelData = ModelSerializer.Save(config);

        // 2. Load model
        var runtime = new RuntimeMobileRuntime();
        var model = runtime.LoadModel(modelData);

        // 3. Prepare input
        var tensorFactory = new TensorFactory();
        var input = tensorFactory.CreateTensor(new[] { -1f, 2f, -3f }, new[] { 3 });

        // 4. Run inference
        var outputs = model.Predict(new[] { input });

        // 5. Verify output
        Assert.IsNotNull(outputs);
        Assert.AreEqual(1, outputs.Length);
        CollectionAssert.AreEqual(
            new[] { 0f, 2f, 0f }, // ReLU applied
            outputs[0].ToArray<float>()
        );
    }

    [TestMethod]
    public void SwitchBackend_CPUtoGPU_Succeeds()
    {
        var runtime = new RuntimeMobileRuntime();
        var config = CreateTestModelConfig();
        var modelData = ModelSerializer.Save(config);
        var model = runtime.LoadModel(modelData);

        // Test CPU backend
        runtime.SetHardwareBackend(BackendType.CPU);
        var cpuOutput = model.Predict(new[] { input });

        // Test GPU backend (if available)
        if (MetalBackendFactory.IsAvailable() || VulkanBackendFactory.IsAvailable())
        {
            runtime.SetHardwareBackend(BackendType.GPU);
            var gpuOutput = model.Predict(new[] { input });

            // Results should match
            CollectionAssert.AreEqual(
                cpuOutput[0].ToArray<float>(),
                gpuOutput[0].ToArray<float>()
            );
        }
    }
}
```

### 7. Performance Tests
```csharp
[TestClass]
public class PerformanceTests
{
    [TestMethod]
    public void Inference_Latency_Under20ms()
    {
        var runtime = new RuntimeMobileRuntime();
        var config = CreateTestModelConfig();
        var modelData = ModelSerializer.Save(config);
        var model = runtime.LoadModel(modelData);
        var input = CreateTestInput();

        var stopwatch = Stopwatch.StartNew();
        model.Predict(new[] { input });
        stopwatch.Stop();

        Assert.IsTrue(
            stopwatch.ElapsedMilliseconds < 20,
            $"Inference took {stopwatch.ElapsedMilliseconds}ms, expected < 20ms"
        );
    }

    [TestMethod]
    public void ModelLoading_Under100ms()
    {
        var config = CreateTestModelConfig();
        var modelData = ModelSerializer.Save(config);
        var runtime = new RuntimeMobileRuntime();

        var stopwatch = Stopwatch.StartNew();
        runtime.LoadModel(modelData);
        stopwatch.Stop();

        Assert.IsTrue(
            stopwatch.ElapsedMilliseconds < 100,
            $"Model loading took {stopwatch.ElapsedMilliseconds}ms, expected < 100ms"
        );
    }

    [TestMethod]
    public void MemoryUsage_Under50MB()
    {
        var runtime = new RuntimeMobileRuntime();
        var config = CreateTestModelConfig();
        var modelData = ModelSerializer.Save(config);
        var model = runtime.LoadModel(modelData);

        var stats = model.GetMemoryStats();

        Assert.IsTrue(
            stats.PeakUsage < 50 * 1024 * 1024,
            $"Memory usage was {stats.PeakUsage / 1024 / 1024}MB, expected < 50MB"
        );
    }
}
```

### 8. Memory Leak Tests
```csharp
[TestClass]
public class MemoryLeakTests
{
    [TestMethod]
    public void MultipleInferences_NoMemoryLeaks()
    {
        var runtime = new RuntimeMobileRuntime();
        var config = CreateTestModelConfig();
        var modelData = ModelSerializer.Save(config);
        var model = runtime.LoadModel(modelData);

        var input = CreateTestInput();

        // Force garbage collection before test
        GC.Collect();
        GC.WaitForPendingFinalizers();
        var initialMemory = GC.GetTotalMemory(true);

        // Run 100 inferences
        for (int i = 0; i < 100; i++)
        {
            var outputs = model.Predict(new[] { input });
            outputs[0].Dispose();
        }

        // Force garbage collection after test
        GC.Collect();
        GC.WaitForPendingFinalizers();
        var finalMemory = GC.GetTotalMemory(true);

        // Memory should not grow by more than 10MB
        var memoryGrowth = finalMemory - initialMemory;
        Assert.IsTrue(
            memoryGrowth < 10 * 1024 * 1024,
            $"Memory grew by {memoryGrowth / 1024 / 1024}MB, possible memory leak"
        );
    }
}
```

## File Structure
```
tests/MobileRuntime.Tests/
├── TensorOperationsTests.cs
├── MemoryPoolTests.cs
├── ModelFormatTests.cs
├── CpuBackendTests.cs
├── MetalBackendTests.cs
├── VulkanBackendTests.cs
├── MobileModelTests.cs
├── IntegrationTests.cs
├── PerformanceTests.cs
└── MemoryLeakTests.cs
```

## Success Criteria
- All tests pass
- Test coverage > 80%
- No memory leaks detected
- Performance targets met
- Cross-platform compatibility verified

## Dependencies
- All previous specs
- MSTest or NUnit framework
- FluentAssertions for better assertions

## Platform Requirements
- Windows: x64 for CPU tests
- macOS: ARM64 for Metal tests
- Linux: ARM64/x64 for Vulkan tests
- Android (optional): for Vulkan tests on real device
- iOS (optional): for Metal tests on real device
