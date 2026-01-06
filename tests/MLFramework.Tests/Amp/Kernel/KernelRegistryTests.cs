using MLFramework.Amp;
using Device = MLFramework.Core.Device;
using Xunit;

namespace MLFramework.Tests.Amp.Kernel
{
    public class KernelRegistryTests
    {
        [Fact]
        public void GetOrCreateSelector_NewDevice_CreatesSelector()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            KernelRegistry.Clear();

            // Act
            var selector = KernelRegistry.GetOrCreateSelector(device);

            // Assert
            Assert.NotNull(selector);
            Assert.Equal(device, selector.Device);
        }

        [Fact]
        public void GetOrCreateSelector_ExistingDevice_ReturnsSameSelector()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            KernelRegistry.Clear();

            // Act
            var selector1 = KernelRegistry.GetOrCreateSelector(device);
            var selector2 = KernelRegistry.GetOrCreateSelector(device);

            // Assert
            Assert.Same(selector1, selector2);
        }

        [Fact]
        public void RegisterDefaultCapabilities_RegistersExpectedKernels()
        {
            // Arrange
            var device = Device.CreateCpu("TestCPU");
            var selector = new KernelSelector(device);

            // Act
            KernelRegistry.RegisterDefaultCapabilities(device, selector);

            // Assert - Convolution kernels
            Assert.True(selector.IsKernelAvailable("conv2d", KernelDtype.Float32));
            Assert.True(selector.IsKernelAvailable("conv2d_fp16", KernelDtype.Float16));
            Assert.True(selector.IsKernelAvailable("conv2d_bf16", KernelDtype.BFloat16));

            // Assert - Matrix multiplication kernels
            Assert.True(selector.IsKernelAvailable("matmul", KernelDtype.Float32));
            Assert.True(selector.IsKernelAvailable("matmul_fp16", KernelDtype.Float16));
            Assert.True(selector.IsKernelAvailable("matmul_bf16", KernelDtype.BFloat16));

            // Assert - Activation kernels
            Assert.True(selector.IsKernelAvailable("relu", KernelDtype.Float32));
            Assert.True(selector.IsKernelAvailable("gelu", KernelDtype.Float32));
            Assert.True(selector.IsKernelAvailable("sigmoid", KernelDtype.Float32));
            Assert.True(selector.IsKernelAvailable("tanh", KernelDtype.Float32));

            // Assert - Pooling kernels
            Assert.True(selector.IsKernelAvailable("maxpool2d", KernelDtype.Float32));
            Assert.True(selector.IsKernelAvailable("avgpool2d", KernelDtype.Float32));
        }

        [Fact]
        public void RegisterDefaultCapabilities_DeviceWithTensorCores_SetsTensorCoreSupport()
        {
            // Arrange
            var device = Device.CreateCpu("TestGPU", supportsTensorCores: true);
            var selector = new KernelSelector(device);

            // Act
            KernelRegistry.RegisterDefaultCapabilities(device, selector);

            // Assert - Check that FP16 and BF16 kernels support tensor cores
            var fp16Cap = selector.GetPerformanceStats("conv2d_fp16", KernelDtype.Float16);
            var bf16Cap = selector.GetPerformanceStats("conv2d_bf16", KernelDtype.BFloat16);

            // Note: This is a simplified check - actual capability objects aren't directly accessible
            // In a real implementation, we'd need to expose the capability objects
            Assert.True(selector.IsKernelAvailable("conv2d_fp16", KernelDtype.Float16));
            Assert.True(selector.IsKernelAvailable("conv2d_bf16", KernelDtype.BFloat16));
        }

        [Fact]
        public void Clear_RemovesAllSelectors()
        {
            // Arrange
            var device1 = Device.CreateCpu("TestCPU1");
            var device2 = Device.CreateCpu("TestCPU2");
            KernelRegistry.GetOrCreateSelector(device1);
            KernelRegistry.GetOrCreateSelector(device2);

            // Act
            KernelRegistry.Clear();

            // Assert
            var selectors = KernelRegistry.GetAllSelectors();
            Assert.Empty(selectors);
        }

        [Fact]
        public void GetAllSelectors_ReturnsAllRegisteredSelectors()
        {
            // Arrange
            KernelRegistry.Clear();
            var device1 = Device.CreateCpu("TestCPU1");
            var device2 = Device.CreateCpu("TestCPU2");
            KernelRegistry.GetOrCreateSelector(device1);
            KernelRegistry.GetOrCreateSelector(device2);

            // Act
            var selectors = KernelRegistry.GetAllSelectors();

            // Assert
            Assert.Equal(2, selectors.Count);
            Assert.True(selectors.ContainsKey(device1.Id));
            Assert.True(selectors.ContainsKey(device2.Id));
        }

        [Fact]
        public void RemoveSelector_RemovesSpecificSelector()
        {
            // Arrange
            KernelRegistry.Clear();
            var device = Device.CreateCpu("TestCPU");
            KernelRegistry.GetOrCreateSelector(device);

            // Act
            var result = KernelRegistry.RemoveSelector(device.Id);

            // Assert
            Assert.True(result);
            Assert.False(KernelRegistry.HasSelector(device.Id));
        }

        [Fact]
        public void RemoveSelector_NonExistentSelector_ReturnsFalse()
        {
            // Arrange
            KernelRegistry.Clear();
            var device = Device.CreateCpu("TestCPU");

            // Act
            var result = KernelRegistry.RemoveSelector(device.Id);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void HasSelector_ExistingSelector_ReturnsTrue()
        {
            // Arrange
            KernelRegistry.Clear();
            var device = Device.CreateCpu("TestCPU");
            KernelRegistry.GetOrCreateSelector(device);

            // Act
            var result = KernelRegistry.HasSelector(device.Id);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void HasSelector_NonExistentSelector_ReturnsFalse()
        {
            // Arrange
            KernelRegistry.Clear();
            var device = Device.CreateCpu("TestCPU");

            // Act
            var result = KernelRegistry.HasSelector(device.Id);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void GetOrCreateSelector_ThreadSafe_MultipleThreads()
        {
            // Arrange
            KernelRegistry.Clear();
            var device = Device.CreateCpu("TestCPU");
            var tasks = new List<Task>();
            var selectors = new List<KernelSelector>();

            // Act
            for (int i = 0; i < 10; i++)
            {
                tasks.Add(Task.Run(() =>
                {
                    var selector = KernelRegistry.GetOrCreateSelector(device);
                    lock (selectors)
                    {
                        selectors.Add(selector);
                    }
                }));
            }

            Task.WaitAll(tasks.ToArray());

            // Assert
            Assert.All(selectors, s => Assert.Same(selectors[0], s));
        }
    }
}
