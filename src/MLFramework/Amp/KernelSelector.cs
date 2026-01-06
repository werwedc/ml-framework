using System.Collections.Concurrent;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using Device = MLFramework.Core.Device;

namespace MLFramework.Amp
{
    /// <summary>
    /// Selects appropriate GPU kernels based on tensor data types
    /// </summary>
    public class KernelSelector
    {
        private readonly Device _device;
        private readonly ConcurrentDictionary<string, Dictionary<KernelDtype, KernelCapability>> _kernelCapabilities;
        private readonly ConcurrentDictionary<string, Dictionary<KernelDtype, KernelPerformanceStats>> _performanceStats;
        private readonly object _lock = new object();

        /// <summary>
        /// Gets the device this selector is for
        /// </summary>
        public Device Device => _device;

        /// <summary>
        /// Creates a new KernelSelector for the given device
        /// </summary>
        /// <param name="device">The GPU device</param>
        public KernelSelector(Device device)
        {
            _device = device;
            _kernelCapabilities = new ConcurrentDictionary<string, Dictionary<KernelDtype, KernelCapability>>();
            _performanceStats = new ConcurrentDictionary<string, Dictionary<KernelDtype, KernelPerformanceStats>>();
        }

        /// <summary>
        /// Gets the kernel dtype for a tensor
        /// </summary>
        /// <param name="tensor">The input tensor</param>
        /// <returns>The kernel dtype</returns>
        public KernelDtype GetKernelDtype(Tensor tensor)
        {
            return MapDataTypeToKernelDtype(tensor.Dtype);
        }

        /// <summary>
        /// Gets the kernel dtype for a list of tensors
        /// </summary>
        /// <param name="tensors">The input tensors</param>
        /// <returns>The kernel dtype</returns>
        public KernelDtype GetKernelDtype(IList<Tensor> tensors)
        {
            if (tensors.Count == 0)
            {
                return KernelDtype.Float32;
            }

            // Check if all tensors have the same dtype
            var firstDtype = tensors[0].Dtype;
            bool allSameDtype = tensors.All(t => t.Dtype == firstDtype);

            if (allSameDtype)
            {
                return MapDataTypeToKernelDtype(firstDtype);
            }
            else
            {
                return KernelDtype.Mixed;
            }
        }

        /// <summary>
        /// Gets the kernel dtype for an operation
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="inputDtypes">The input tensor data types</param>
        /// <returns>The kernel dtype</returns>
        public KernelDtype GetKernelDtype(
            string operationName,
            IList<DataType> inputDtypes)
        {
            if (inputDtypes.Count == 0)
            {
                return KernelDtype.Float32;
            }

            // Check if all dtypes are the same
            var firstDtype = inputDtypes[0];
            bool allSameDtype = inputDtypes.All(d => d == firstDtype);

            if (allSameDtype)
            {
                return MapDataTypeToKernelDtype(firstDtype);
            }
            else
            {
                return KernelDtype.Mixed;
            }
        }

        /// <summary>
        /// Checks if a kernel is available for the given dtype
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="dtype">The kernel data type</param>
        /// <returns>True if available, false otherwise</returns>
        public bool IsKernelAvailable(string operationName, KernelDtype dtype)
        {
            if (!_kernelCapabilities.TryGetValue(operationName, out var capabilities))
            {
                return false;
            }

            if (!capabilities.TryGetValue(dtype, out var capability))
            {
                return false;
            }

            return capability.IsAvailable;
        }

        /// <summary>
        /// Registers a kernel capability
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="capability">The kernel capability</param>
        public void RegisterKernelCapability(
            string operationName,
            KernelCapability capability)
        {
            var capabilities = _kernelCapabilities.GetOrAdd(operationName, _ => new Dictionary<KernelDtype, KernelCapability>());

            lock (_lock)
            {
                capabilities[capability.Dtype] = capability;
            }
        }

        /// <summary>
        /// Gets the best available kernel dtype for an operation
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="inputDtypes">The input tensor data types</param>
        /// <param name="preferredDtype">The preferred dtype (optional)</param>
        /// <returns>The best available kernel dtype</returns>
        public KernelDtype SelectBestKernel(
            string operationName,
            IList<DataType> inputDtypes,
            KernelDtype? preferredDtype = null)
        {
            // If preferred dtype is available and valid, use it
            if (preferredDtype.HasValue && IsKernelAvailable(operationName, preferredDtype.Value))
            {
                return preferredDtype.Value;
            }

            // Try to match input dtype
            foreach (var inputDtype in inputDtypes)
            {
                var kernelDtype = MapDataTypeToKernelDtype(inputDtype);
                if (IsKernelAvailable(operationName, kernelDtype))
                {
                    return kernelDtype;
                }
            }

            // Fall back to FP32 (always available)
            return KernelDtype.Float32;
        }

        /// <summary>
        /// Gets kernel performance statistics
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="dtype">The kernel data type</param>
        /// <returns>Performance statistics if available</returns>
        public KernelPerformanceStats? GetPerformanceStats(
            string operationName,
            KernelDtype dtype)
        {
            if (!_performanceStats.TryGetValue(operationName, out var stats))
            {
                return null;
            }

            if (!stats.TryGetValue(dtype, out var stat))
            {
                return null;
            }

            return stat;
        }

        /// <summary>
        /// Updates kernel performance statistics
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="dtype">The kernel data type</param>
        /// <param name="executionTime">The execution time in milliseconds</param>
        public void UpdatePerformanceStats(
            string operationName,
            KernelDtype dtype,
            float executionTime)
        {
            var stats = _performanceStats.GetOrAdd(operationName, _ => new Dictionary<KernelDtype, KernelPerformanceStats>());

            lock (_lock)
            {
                if (stats.TryGetValue(dtype, out var existingStats))
                {
                    // Update existing stats
                    var newAvg = (existingStats.AverageExecutionTime * existingStats.ExecutionCount + executionTime) /
                                 (existingStats.ExecutionCount + 1);
                    var newMin = Math.Min(existingStats.MinExecutionTime, executionTime);
                    var newMax = Math.Max(existingStats.MaxExecutionTime, executionTime);
                    var newCount = existingStats.ExecutionCount + 1;

                    var updatedStats = new KernelPerformanceStats(
                        operationName,
                        dtype,
                        newAvg,
                        newMin,
                        newMax,
                        newCount);

                    stats[dtype] = updatedStats;
                }
                else
                {
                    // Create new stats entry
                    var newStats = new KernelPerformanceStats(
                        operationName,
                        dtype,
                        executionTime,
                        executionTime,
                        executionTime,
                        1);

                    stats[dtype] = newStats;
                }
            }
        }

        /// <summary>
        /// Maps DataType to KernelDtype
        /// </summary>
        private KernelDtype MapDataTypeToKernelDtype(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => KernelDtype.Float16,
                DataType.BFloat16 => KernelDtype.BFloat16,
                DataType.Float32 => KernelDtype.Float32,
                _ => KernelDtype.Float32
            };
        }
    }
}
