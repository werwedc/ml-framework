# Spec: AMP Kernel Selector

## Overview
Implement a kernel selection system that chooses the appropriate GPU kernel based on the tensor data type (FP32, FP16, BF16) for optimal performance in Automatic Mixed Precision training.

## Class Specification

### 1. KernelDtype Enum

**File:** `src/MLFramework/Amp/KernelDtype.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Kernel data type for GPU operations
    /// </summary>
    public enum KernelDtype
    {
        /// <summary>
        /// Float32 (default precision)
        /// </summary>
        Float32 = 0,

        /// <summary>
        /// Float16 (half precision)
        /// </summary>
        Float16 = 1,

        /// <summary>
        /// BFloat16 (brain float)
        /// </summary>
        BFloat16 = 2,

        /// <summary>
        /// Mixed precision (multiple dtypes)
        /// </summary>
        Mixed = 3,

        /// <summary>
        /// Automatic selection based on tensor dtype
        /// </summary>
        Auto = 4
    }
}
```

### 2. KernelSelector Class

**File:** `src/MLFramework/Amp/KernelSelector.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Selects appropriate GPU kernels based on tensor data types
    /// </summary>
    public class KernelSelector
    {
        private readonly Device _device;
        private readonly Dictionary<string, KernelCapability> _kernelCapabilities;
        private readonly object _lock = new object();

        /// <summary>
        /// Gets the device this selector is for
        /// </summary>
        public Device Device => _device;

        /// <summary>
        /// Creates a new KernelSelector for the given device
        /// </summary>
        /// <param name="device">The GPU device</param>
        public KernelSelector(Device device);

        /// <summary>
        /// Gets the kernel dtype for a tensor
        /// </summary>
        /// <param name="tensor">The input tensor</param>
        /// <returns>The kernel dtype</returns>
        public KernelDtype GetKernelDtype(Tensor tensor);

        /// <summary>
        /// Gets the kernel dtype for a list of tensors
        /// </summary>
        /// <param name="tensors">The input tensors</param>
        /// <returns>The kernel dtype</returns>
        public KernelDtype GetKernelDtype(IList<Tensor> tensors);

        /// <summary>
        /// Gets the kernel dtype for an operation
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="inputDtypes">The input tensor data types</param>
        /// <returns>The kernel dtype</returns>
        public KernelDtype GetKernelDtype(
            string operationName,
            IList<DataType> inputDtypes);

        /// <summary>
        /// Checks if a kernel is available for the given dtype
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="dtype">The kernel data type</param>
        /// <returns>True if available, false otherwise</returns>
        public bool IsKernelAvailable(string operationName, KernelDtype dtype);

        /// <summary>
        /// Registers a kernel capability
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="capability">The kernel capability</param>
        public void RegisterKernelCapability(
            string operationName,
            KernelCapability capability);

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
            KernelDtype? preferredDtype = null);

        /// <summary>
        /// Gets kernel performance statistics
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="dtype">The kernel data type</param>
        /// <returns>Performance statistics if available</returns>
        public KernelPerformanceStats? GetPerformanceStats(
            string operationName,
            KernelDtype dtype);

        /// <summary>
        /// Updates kernel performance statistics
        /// </summary>
        /// <param name="operationName">The name of the operation</param>
        /// <param name="dtype">The kernel data type</param>
        /// <param name="executionTime">The execution time in milliseconds</param>
        public void UpdatePerformanceStats(
            string operationName,
            KernelDtype dtype,
            float executionTime);
    }
}
```

### 3. KernelCapability Class

**File:** `src/MLFramework/Amp/KernelCapability.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Describes the capabilities of a GPU kernel
    /// </summary>
    public class KernelCapability
    {
        /// <summary>
        /// Gets the kernel data type
        /// </summary>
        public KernelDtype Dtype { get; }

        /// <summary>
        /// Gets whether the kernel is available on the device
        /// </summary>
        public bool IsAvailable { get; }

        /// <summary>
        /// Gets whether the kernel supports tensor cores (if applicable)
        /// </summary>
        public bool SupportsTensorCores { get; }

        /// <summary>
        /// Gets the relative performance factor (higher = faster)
        /// </summary>
        public float PerformanceFactor { get; }

        /// <summary>
        /// Gets the memory efficiency factor (higher = more memory efficient)
        /// </summary>
        public float MemoryEfficiency { get; }

        /// <summary>
        /// Creates a new KernelCapability
        /// </summary>
        public KernelCapability(
            KernelDtype dtype,
            bool isAvailable = true,
            bool supportsTensorCores = false,
            float performanceFactor = 1.0f,
            float memoryEfficiency = 1.0f);

        /// <summary>
        /// Creates a kernel capability for FP32
        /// </summary>
        public static KernelCapability CreateFloat32(
            bool supportsTensorCores = false);

        /// <summary>
        /// Creates a kernel capability for FP16
        /// </summary>
        public static KernelCapability CreateFloat16(
            bool supportsTensorCores = true);

        /// <summary>
        /// Creates a kernel capability for BF16
        /// </summary>
        public static KernelCapability CreateBFloat16(
            bool supportsTensorCores = true);
    }
}
```

### 4. KernelPerformanceStats Class

**File:** `src/MLFramework/Amp/KernelPerformanceStats.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Performance statistics for a GPU kernel
    /// </summary>
    public class KernelPerformanceStats
    {
        /// <summary>
        /// Gets the operation name
        /// </summary>
        public string OperationName { get; }

        /// <summary>
        /// Gets the kernel data type
        /// </summary>
        public KernelDtype Dtype { get; }

        /// <summary>
        /// Gets the average execution time (ms)
        /// </summary>
        public float AverageExecutionTime { get; }

        /// <summary>
        /// Gets the minimum execution time (ms)
        /// </summary>
        public float MinExecutionTime { get; }

        /// <summary>
        /// Gets the maximum execution time (ms)
        /// </summary>
        public float MaxExecutionTime { get; }

        /// <summary>
        /// Gets the number of executions
        /// </summary>
        public int ExecutionCount { get; }

        /// <summary>
        /// Creates a new KernelPerformanceStats
        /// </summary>
        public KernelPerformanceStats(
            string operationName,
            KernelDtype dtype,
            float averageExecutionTime,
            float minExecutionTime,
            float maxExecutionTime,
            int executionCount);

        /// <summary>
        /// Returns a string representation of the statistics
        /// </summary>
        public override string ToString();
    }
}
```

### 5. KernelRegistry Class

**File:** `src/MLFramework/Amp/KernelRegistry.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Global registry for kernel capabilities across devices
    /// </summary>
    public static class KernelRegistry
    {
        private static readonly Dictionary<DeviceId, KernelSelector> _selectors;
        private static readonly object _lock = new object();

        /// <summary>
        /// Gets or creates a KernelSelector for the given device
        /// </summary>
        /// <param name="device">The GPU device</param>
        /// <returns>The KernelSelector</returns>
        public static KernelSelector GetOrCreateSelector(Device device);

        /// <summary>
        /// Registers default kernel capabilities for a device
        /// </summary>
        /// <param name="device">The GPU device</param>
        /// <param name="selector">The KernelSelector to populate</param>
        public static void RegisterDefaultCapabilities(Device device, KernelSelector selector);

        /// <summary>
        /// Clears all registered kernel capabilities
        /// </summary>
        public static void Clear();

        /// <summary>
        /// Gets all registered selectors
        /// </summary>
        /// <returns>Dictionary of device IDs to selectors</returns>
        public static IReadOnlyDictionary<DeviceId, KernelSelector> GetAllSelectors();
    }
}
```

## Implementation Details

### Kernel Selection Logic

```csharp
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
```

### Best Kernel Selection

```csharp
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
```

### Default Kernel Capabilities

```csharp
public static void RegisterDefaultCapabilities(Device device, KernelSelector selector)
{
    // Convolution kernels
    selector.RegisterKernelCapability("conv2d", KernelCapability.CreateFloat32());
    selector.RegisterKernelCapability("conv2d_fp16", KernelCapability.CreateFloat16(true));
    selector.RegisterKernelCapability("conv2d_bf16", KernelCapability.CreateBFloat16(true));

    // Matrix multiplication kernels
    selector.RegisterKernelCapability("matmul", KernelCapability.CreateFloat32());
    selector.RegisterKernelCapability("matmul_fp16", KernelCapability.CreateFloat16(true));
    selector.RegisterKernelCapability("matmul_bf16", KernelCapability.CreateBFloat16(true));

    // Activation kernels
    selector.RegisterKernelCapability("relu", KernelCapability.CreateFloat32());
    selector.RegisterKernelCapability("gelu", KernelCapability.CreateFloat32());

    // Pooling kernels
    selector.RegisterKernelCapability("maxpool2d", KernelCapability.CreateFloat32());
    selector.RegisterKernelCapability("avgpool2d", KernelCapability.CreateFloat32());
}
```

### Performance Tracking

```csharp
public void UpdatePerformanceStats(
    string operationName,
    KernelDtype dtype,
    float executionTime)
{
    lock (_lock)
    {
        var stats = GetPerformanceStats(operationName, dtype);
        if (stats == null)
        {
            // Create new stats entry
            var newStats = new KernelPerformanceStats(
                operationName,
                dtype,
                executionTime,
                executionTime,
                executionTime,
                1);
            // Store stats...
        }
        else
        {
            // Update existing stats
            var newAvg = (stats.AverageExecutionTime * stats.ExecutionCount + executionTime) /
                        (stats.ExecutionCount + 1);
            var newMin = Math.Min(stats.MinExecutionTime, executionTime);
            var newMax = Math.Max(stats.MaxExecutionTime, executionTime);
            // Update stored stats...
        }
    }
}
```

## Usage Examples

### Basic Kernel Selection
```csharp
var selector = KernelRegistry.GetOrCreateSelector(device);

// Get kernel dtype for a tensor
var kernelDtype = selector.GetKernelDtype(tensor);

// Get kernel dtype for an operation
var kernelDtype = selector.GetKernelDtype("conv2d", new[] { DataType.Float16 });
```

### Register Custom Kernel
```csharp
var capability = new KernelCapability(
    dtype: KernelDtype.Float16,
    isAvailable: true,
    supportsTensorCores: true,
    performanceFactor: 2.0f, // 2x faster than FP32
    memoryEfficiency: 2.0f);  // 2x more memory efficient

selector.RegisterKernelCapability("custom_op", capability);
```

### Check Kernel Availability
```csharp
bool hasFp16Kernel = selector.IsKernelAvailable("conv2d", KernelDtype.Float16);
bool hasBf16Kernel = selector.IsKernelAvailable("matmul", KernelDtype.BFloat16);
```

### Get Performance Stats
```csharp
var stats = selector.GetPerformanceStats("conv2d", KernelDtype.Float16);
if (stats != null)
{
    Console.WriteLine($"Avg time: {stats.AverageExecutionTime}ms");
    Console.WriteLine($"Executions: {stats.ExecutionCount}");
}
```

## Dependencies
- MLFramework.Core (Tensor, DataType, Device, DeviceId)
- System.Collections.Generic (Dictionary, List, IReadOnlyDictionary)
- System.Linq (for LINQ operations)
- System (Math)

## Testing Requirements
- Test kernel dtype selection for various tensor dtypes
- Test kernel availability checking
- Test best kernel selection logic
- Test kernel capability registration
- Test performance stats tracking
- Test default capability registration
- Test mixed dtype handling
- Test thread safety of kernel registry

## Success Criteria
- [ ] Kernel dtype selection works correctly
- [ ] Kernel availability checks are accurate
- [ ] Best kernel selection matches preferred dtype when available
- [ ] Default capabilities are registered correctly
- [ ] Performance stats are tracked accurately
- [ ] Thread-safe operations verified
- [ ] Performance overhead < 2% of kernel selection time
- [ ] All unit tests pass
- [ ] Documentation includes usage examples
