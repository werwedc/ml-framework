using System.Collections.Concurrent;
using MLFramework.Core;

namespace MLFramework.Amp
{
    /// <summary>
    /// Global registry for kernel capabilities across devices
    /// </summary>
    public static class KernelRegistry
    {
        private static readonly ConcurrentDictionary<DeviceId, KernelSelector> _selectors;
        private static readonly object _lock = new object();

        static KernelRegistry()
        {
            _selectors = new ConcurrentDictionary<DeviceId, KernelSelector>();
        }

        /// <summary>
        /// Gets or creates a KernelSelector for the given device
        /// </summary>
        /// <param name="device">The GPU device</param>
        /// <returns>The KernelSelector</returns>
        public static KernelSelector GetOrCreateSelector(Device device)
        {
            return _selectors.GetOrAdd(device.Id, id =>
            {
                var selector = new KernelSelector(device);
                RegisterDefaultCapabilities(device, selector);
                return selector;
            });
        }

        /// <summary>
        /// Registers default kernel capabilities for a device
        /// </summary>
        /// <param name="device">The GPU device</param>
        /// <param name="selector">The KernelSelector to populate</param>
        public static void RegisterDefaultCapabilities(Device device, KernelSelector selector)
        {
            // Convolution kernels
            selector.RegisterKernelCapability("conv2d", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("conv2d_fp16", KernelCapability.CreateFloat16(device.SupportsTensorCores));
            selector.RegisterKernelCapability("conv2d_bf16", KernelCapability.CreateBFloat16(device.SupportsTensorCores));

            // Matrix multiplication kernels
            selector.RegisterKernelCapability("matmul", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("matmul_fp16", KernelCapability.CreateFloat16(device.SupportsTensorCores));
            selector.RegisterKernelCapability("matmul_bf16", KernelCapability.CreateBFloat16(device.SupportsTensorCores));

            // Activation kernels
            selector.RegisterKernelCapability("relu", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("gelu", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("sigmoid", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("tanh", KernelCapability.CreateFloat32());

            // Pooling kernels
            selector.RegisterKernelCapability("maxpool2d", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("avgpool2d", KernelCapability.CreateFloat32());

            // Normalization kernels
            selector.RegisterKernelCapability("batchnorm", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("layernorm", KernelCapability.CreateFloat32());

            // Reduction kernels
            selector.RegisterKernelCapability("sum", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("mean", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("max", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("min", KernelCapability.CreateFloat32());

            // Element-wise operations
            selector.RegisterKernelCapability("add", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("sub", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("mul", KernelCapability.CreateFloat32());
            selector.RegisterKernelCapability("div", KernelCapability.CreateFloat32());
        }

        /// <summary>
        /// Clears all registered kernel capabilities
        /// </summary>
        public static void Clear()
        {
            lock (_lock)
            {
                _selectors.Clear();
            }
        }

        /// <summary>
        /// Gets all registered selectors
        /// </summary>
        /// <returns>Dictionary of device IDs to selectors</returns>
        public static IReadOnlyDictionary<DeviceId, KernelSelector> GetAllSelectors()
        {
            return _selectors.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }

        /// <summary>
        /// Removes a specific selector from the registry
        /// </summary>
        /// <param name="deviceId">The device ID</param>
        /// <returns>True if removed, false otherwise</returns>
        public static bool RemoveSelector(DeviceId deviceId)
        {
            return _selectors.TryRemove(deviceId, out _);
        }

        /// <summary>
        /// Checks if a selector exists for the given device
        /// </summary>
        /// <param name="deviceId">The device ID</param>
        /// <returns>True if exists, false otherwise</returns>
        public static bool HasSelector(DeviceId deviceId)
        {
            return _selectors.ContainsKey(deviceId);
        }
    }
}
