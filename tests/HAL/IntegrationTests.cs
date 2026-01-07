using NUnit.Framework;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.HAL.CUDA;

namespace MLFramework.Tests.HAL;

/// <summary>
/// Integration tests for the HAL system
/// Tests full workflows across components including device transfers, backend operations, and memory management
/// </summary>
[TestFixture]
public class IntegrationTests
{
    [SetUp]
    public void Setup()
    {
        // Clear registry before each test to ensure clean state
        BackendRegistry.Clear();
    }

    [Test]
    public void FullWorkflow_CpuOnly()
    {
        // Register CPU backend
        BackendRegistry.Register(new CpuBackend());

        // Create tensors on CPU
        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
        var b = TensorHALExtensions.FromArray(new[] { 4.0f, 5.0f, 6.0f }, Device.CPU);

        // Perform operations using backend
        var backend = BackendRegistry.GetBackend(DeviceType.CPU);
        Assert.IsNotNull(backend);

        var addResult = backend.ExecuteOperation(Operation.Add, new[] { a, b });
        var reluResult = backend.ExecuteOperation(Operation.ReLU, new[] { addResult });

        // Verify results
        var expected = new[] { 5.0f, 7.0f, 9.0f };
        CollectionAssert.AreEqual(expected, reluResult.ToArray(), new FloatComparer(0.001f));
    }

    [Test]
    public void DeviceTransfer_CpuToCpu()
    {
        BackendRegistry.Register(new CpuBackend());

        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
        var b = a.To(Device.CPU);

        // CPU to CPU should return same tensor (no-op)
        Assert.AreSame(a, b);
        CollectionAssert.AreEqual(a.ToArray(), b.ToArray());
    }

    [Test]
    public void DeviceTransfer_CpuToGpu()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        BackendRegistry.Register(new CpuBackend());
        BackendRegistry.Register(new CudaBackend());

        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
        var b = a.To(Device.CUDA(0));

        // Verify tensor was moved to CUDA
        Assert.IsNotNull(b);
        Assert.AreEqual(3, b.Size);

        // Copy back to CPU and verify data integrity
        var c = b.To(Device.CPU);
        CollectionAssert.AreEqual(a.ToArray(), c.ToArray());
    }

    [Test]
    public void BackendRegistry_GetAvailableDevices()
    {
        BackendRegistry.Clear();
        BackendRegistry.Register(new CpuBackend());

        if (CudaAvailable())
        {
            BackendRegistry.Register(new CudaBackend());
        }

        var devices = BackendRegistry.GetAvailableDevices().ToList();

        Assert.Contains(DeviceType.CPU, devices);

        if (CudaAvailable())
        {
            Assert.Contains(DeviceType.CUDA, devices);
        }
    }

    [Test]
    public void BackendRegistry_GetBackend_ReturnsCorrectBackend()
    {
        BackendRegistry.Register(new CpuBackend());

        var backend = BackendRegistry.GetBackend(DeviceType.CPU);

        Assert.IsNotNull(backend);
        Assert.AreEqual(DeviceType.CPU, backend.Type);
        Assert.AreEqual("ManagedCPU", backend.Name);
    }

    [Test]
    public void BackendRegistry_IsDeviceAvailable_WorksCorrectly()
    {
        BackendRegistry.Register(new CpuBackend());

        Assert.IsTrue(BackendRegistry.IsDeviceAvailable(DeviceType.CPU));
        Assert.IsFalse(BackendRegistry.IsDeviceAvailable(DeviceType.CUDA));
    }

    [Test]
    public void CpuBackend_MultipleOperations_Workflow()
    {
        BackendRegistry.Register(new CpuBackend());

        // Create tensors
        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
        var b = TensorHALExtensions.FromArray(new[] { 2.0f, 3.0f, 4.0f }, Device.CPU);

        var backend = BackendRegistry.GetBackend(DeviceType.CPU);

        // Chain multiple operations
        var add = backend.ExecuteOperation(Operation.Add, new[] { a, b });      // [3, 5, 7]
        var mul = backend.ExecuteOperation(Operation.Multiply, new[] { add, b }); // [6, 15, 28]
        var sum = backend.ExecuteOperation(Operation.Sum, new[] { mul });        // [49]

        Assert.AreEqual(49.0f, sum.ToArray()[0], 0.001f);
    }

    [Test]
    public void CpuBackend_ActivationFunctions_Workflow()
    {
        BackendRegistry.Register(new CpuBackend());

        var a = TensorHALExtensions.FromArray(new[] { -1.0f, 0.0f, 1.0f }, Device.CPU);
        var backend = BackendRegistry.GetBackend(DeviceType.CPU);

        var relu = backend.ExecuteOperation(Operation.ReLU, new[] { a });
        var sigmoid = backend.ExecuteOperation(Operation.Sigmoid, new[] { relu });
        var tanh = backend.ExecuteOperation(Operation.Tanh, new[] { a });

        // Verify ReLU: [0, 0, 1]
        var expectedRelu = new[] { 0.0f, 0.0f, 1.0f };
        CollectionAssert.AreEqual(expectedRelu, relu.ToArray(), new FloatComparer(0.001f));

        // Verify tanh(0) = 0
        Assert.AreEqual(0.0f, tanh.ToArray()[1], 0.001f);
    }

    [Test]
    public void CpuBackend_Reductions_Workflow()
    {
        BackendRegistry.Register(new CpuBackend());

        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }, Device.CPU);
        var backend = BackendRegistry.GetBackend(DeviceType.CPU);

        var sum = backend.ExecuteOperation(Operation.Sum, new[] { a });
        var mean = backend.ExecuteOperation(Operation.Mean, new[] { a });

        // Sum = 15, Mean = 3
        Assert.AreEqual(15.0f, sum.ToArray()[0], 0.001f);
        Assert.AreEqual(3.0f, mean.ToArray()[0], 0.001f);
    }

    [Test]
    public void TensorFromAndToArray_Roundtrip()
    {
        BackendRegistry.Register(new CpuBackend());

        var expected = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var tensor = TensorHALExtensions.FromArray(expected, Device.CPU);
        var result = tensor.ToArray();

        CollectionAssert.AreEqual(expected, result, new FloatComparer(0.001f));
    }

    [Test]
    public void TensorZeros_CreatesCorrectTensor()
    {
        BackendRegistry.Register(new CpuBackend());

        var tensor = TensorHALExtensions.Zeros(new[] { 2, 3 }, Device.CPU);

        Assert.AreEqual(new[] { 2, 3 }, tensor.Shape);
        Assert.AreEqual(6, tensor.Size);

        var data = tensor.ToArray();
        foreach (var value in data)
        {
            Assert.AreEqual(0.0f, value, 0.001f);
        }
    }

    private bool CudaAvailable()
    {
        try
        {
            var result = CudaApi.CudaGetDeviceCount(out int count);
            return result == CudaError.Success && count > 0;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
    }

    /// <summary>
    /// Helper class for comparing floating point arrays with tolerance
    /// </summary>
    private class FloatComparer : IComparer
    {
        private readonly float _tolerance;

        public FloatComparer(float tolerance = 0.0001f)
        {
            _tolerance = tolerance;
        }

        public int Compare(object? x, object? y)
        {
            var a = (float)x!;
            var b = (float)y!;
            return Math.Abs(a - b) < _tolerance ? 0 : a.CompareTo(b);
        }
    }
}
