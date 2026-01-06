using MLFramework.Amp;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using Device = MLFramework.Core.Device;
using Xunit;

namespace MLFramework.Tests.Amp.Kernel
{
    public class KernelSelectorTests
    {
        [Fact]
        public void GetKernelDtype_SingleTensor_ReturnsCorrectDtype()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var tensorFP32 = new Tensor(new float[10], new[] { 10 }, false, DataType.Float32);
            var tensorFP16 = new Tensor(new float[10], new[] { 10 }, false, DataType.Float16);
            var tensorBF16 = new Tensor(new float[10], new[] { 10 }, false, DataType.BFloat16);

            // Act & Assert
            Assert.Equal(KernelDtype.Float32, selector.GetKernelDtype(tensorFP32));
            Assert.Equal(KernelDtype.Float16, selector.GetKernelDtype(tensorFP16));
            Assert.Equal(KernelDtype.BFloat16, selector.GetKernelDtype(tensorBF16));
        }

        [Fact]
        public void GetKernelDtype_MultipleTensorsSameDtype_ReturnsCorrectDtype()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var tensors = new List<Tensor>
            {
                new Tensor(new float[10], new[] { 10 }, false, DataType.Float16),
                new Tensor(new float[10], new[] { 10 }, false, DataType.Float16),
                new Tensor(new float[10], new[] { 10 }, false, DataType.Float16)
            };

            // Act
            var result = selector.GetKernelDtype(tensors);

            // Assert
            Assert.Equal(KernelDtype.Float16, result);
        }

        [Fact]
        public void GetKernelDtype_MultipleTensorsMixedDtype_ReturnsMixed()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var tensors = new List<Tensor>
            {
                new Tensor(new float[10], new[] { 10 }, false, DataType.Float16),
                new Tensor(new float[10], new[] { 10 }, false, DataType.Float32),
                new Tensor(new float[10], new[] { 10 }, false, DataType.BFloat16)
            };

            // Act
            var result = selector.GetKernelDtype(tensors);

            // Assert
            Assert.Equal(KernelDtype.Mixed, result);
        }

        [Fact]
        public void GetKernelDtype_EmptyTensorList_ReturnsFloat32()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var tensors = new List<Tensor>();

            // Act
            var result = selector.GetKernelDtype(tensors);

            // Assert
            Assert.Equal(KernelDtype.Float32, result);
        }

        [Fact]
        public void GetKernelDtype_OperationWithInputDtypes_SameDtype_ReturnsCorrectDtype()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var dtypes = new List<DataType> { DataType.Float16, DataType.Float16, DataType.Float16 };

            // Act
            var result = selector.GetKernelDtype("conv2d", dtypes);

            // Assert
            Assert.Equal(KernelDtype.Float16, result);
        }

        [Fact]
        public void GetKernelDtype_OperationWithInputDtypes_MixedDtype_ReturnsMixed()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var dtypes = new List<DataType> { DataType.Float16, DataType.Float32, DataType.BFloat16 };

            // Act
            var result = selector.GetKernelDtype("matmul", dtypes);

            // Assert
            Assert.Equal(KernelDtype.Mixed, result);
        }

        [Fact]
        public void RegisterKernelCapability_RegistersCapability()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var capability = new KernelCapability(KernelDtype.Float16, true, true, 2.0f, 2.0f);

            // Act
            selector.RegisterKernelCapability("custom_op", capability);

            // Assert
            Assert.True(selector.IsKernelAvailable("custom_op", KernelDtype.Float16));
        }

        [Fact]
        public void IsKernelAvailable_RegisteredKernel_ReturnsTrue()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var capability = KernelCapability.CreateFloat16(true);
            selector.RegisterKernelCapability("conv2d", capability);

            // Act
            var result = selector.IsKernelAvailable("conv2d", KernelDtype.Float16);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void IsKernelAvailable_UnregisteredKernel_ReturnsFalse()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);

            // Act
            var result = selector.IsKernelAvailable("unknown_op", KernelDtype.Float16);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void IsKernelAvailable_UnavailableKernel_ReturnsFalse()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var capability = new KernelCapability(KernelDtype.Float16, isAvailable: false);
            selector.RegisterKernelCapability("conv2d", capability);

            // Act
            var result = selector.IsKernelAvailable("conv2d", KernelDtype.Float16);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void SelectBestKernel_WithPreferredAvailable_ReturnsPreferred()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            selector.RegisterKernelCapability("matmul", KernelCapability.CreateFloat16());
            selector.RegisterKernelCapability("matmul", KernelCapability.CreateFloat32());
            var inputDtypes = new List<DataType> { DataType.Float16 };

            // Act
            var result = selector.SelectBestKernel("matmul", inputDtypes, KernelDtype.Float16);

            // Assert
            Assert.Equal(KernelDtype.Float16, result);
        }

        [Fact]
        public void SelectBestKernel_PreferredUnavailable_ReturnsInputDtype()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            selector.RegisterKernelCapability("matmul", KernelCapability.CreateFloat32());
            var inputDtypes = new List<DataType> { DataType.Float16 };

            // Act
            var result = selector.SelectBestKernel("matmul", inputDtypes, KernelDtype.BFloat16);

            // Assert
            Assert.Equal(KernelDtype.Float32, result);
        }

        [Fact]
        public void SelectBestKernel_NoMatchingKernel_ReturnsFloat32()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            selector.RegisterKernelCapability("matmul", KernelCapability.CreateFloat32());
            var inputDtypes = new List<DataType> { DataType.Int32 };

            // Act
            var result = selector.SelectBestKernel("matmul", inputDtypes);

            // Assert
            Assert.Equal(KernelDtype.Float32, result);
        }

        [Fact]
        public void GetPerformanceStats_NoStats_ReturnsNull()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);

            // Act
            var result = selector.GetPerformanceStats("conv2d", KernelDtype.Float16);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void UpdatePerformanceStats_NewStats_CreatesStatsEntry()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);

            // Act
            selector.UpdatePerformanceStats("conv2d", KernelDtype.Float16, 10.5f);
            var result = selector.GetPerformanceStats("conv2d", KernelDtype.Float16);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(10.5f, result!.AverageExecutionTime);
            Assert.Equal(10.5f, result!.MinExecutionTime);
            Assert.Equal(10.5f, result!.MaxExecutionTime);
            Assert.Equal(1, result!.ExecutionCount);
        }

        [Fact]
        public void UpdatePerformanceStats_ExistingStats_UpdatesStats()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);

            // Act
            selector.UpdatePerformanceStats("conv2d", KernelDtype.Float16, 10.0f);
            selector.UpdatePerformanceStats("conv2d", KernelDtype.Float16, 20.0f);
            selector.UpdatePerformanceStats("conv2d", KernelDtype.Float16, 15.0f);
            var result = selector.GetPerformanceStats("conv2d", KernelDtype.Float16);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(15.0f, result!.AverageExecutionTime);
            Assert.Equal(10.0f, result!.MinExecutionTime);
            Assert.Equal(20.0f, result!.MaxExecutionTime);
            Assert.Equal(3, result!.ExecutionCount);
        }

        [Fact]
        public void UpdatePerformanceStats_ThreadSafe_MultipleThreads()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);
            var tasks = new List<Task>();

            // Act
            for (int i = 0; i < 100; i++)
            {
                int index = i;
                tasks.Add(Task.Run(() => selector.UpdatePerformanceStats("conv2d", KernelDtype.Float16, index * 1.0f)));
            }

            Task.WaitAll(tasks.ToArray());
            var result = selector.GetPerformanceStats("conv2d", KernelDtype.Float16);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(100, result!.ExecutionCount);
        }
    }
}
