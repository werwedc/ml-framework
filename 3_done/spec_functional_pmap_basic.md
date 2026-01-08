# Spec: Basic Parallelization (pmap)

## Overview
Implement the pmap transformation for Single Program Multiple Data (SPMD) execution across devices. This will distribute computation across a device mesh.

## Scope
- Implement Parallelize method
- Support 1D device meshes
- Basic data sharding across devices
- Result gathering from all devices

## Technical Requirements

### 1. Parallelize Extension Method

```csharp
namespace MLFramework.Functional
{
    public static class Functional
    {
        /// <summary>
        /// Parallelizes a function across a device mesh (SPMD).
        /// </summary>
        /// <param name="func">Function to parallelize</param>
        /// <param name="mesh">Device mesh for distribution</param>
        /// <param name="in_axes">Which axes to shard across devices (default: data axis)</param>
        /// <returns>Parallel function that returns sharded result</returns>
        public static Func<Tensor, Tensor> Parallelize(
            Func<Tensor, Tensor> func,
            DeviceMesh mesh,
            object[] in_axes = null)
        {
            var transform = new PMapTransform(func, mesh, in_axes);
            return (Func<Tensor, Tensor>)transform.Transform(func);
        }

        /// <summary>
        /// Parallelizes a function with multiple inputs.
        /// </summary>
        public static Func<Tensor, Tensor, Tensor> Parallelize(
            Func<Tensor, Tensor, Tensor> func,
            DeviceMesh mesh,
            object[] in_axes = null)
        {
            var transform = new PMapTransform(func, mesh, in_axes);
            return (Func<Tensor, Tensor, Tensor>)transform.Transform(func);
        }
    }
}
```

### 2. PMapTransform Implementation

```csharp
namespace MLFramework.Functional.Distributed
{
    public class PMapTransform : BaseTransformation
    {
        private readonly DeviceMesh _mesh;
        private readonly int[] _shardAxes;  // -1 means don't shard

        public PMapTransform(Delegate original, DeviceMesh mesh, object[] in_axes = null)
            : base("pmap", TransformationType.Parallelization)
        {
            _mesh = mesh ?? throw new ArgumentNullException(nameof(mesh));
            ValidateDelegate(original);

            // Normalize in_axes
            var paramCount = original.Method.GetParameters().Length;
            _shardAxes = new int[paramCount];

            if (in_axes == null)
            {
                // Default: shard first parameter on axis 0
                _shardAxes[0] = 0;
                for (int i = 1; i < paramCount; i++)
                    _shardAxes[i] = -1;
            }
            else
            {
                if (in_axes.Length != paramCount)
                    throw new ArgumentException($"in_axes length must match parameter count");

                for (int i = 0; i < paramCount; i++)
                {
                    if (in_axes[i] == null)
                        _shardAxes[i] = -1;
                    else if (in_axes[i] is int axis)
                        _shardAxes[i] = axis;
                    else
                        throw new ArgumentException($"in_axes[{i}] must be int or null");
                }
            }
        }

        public override Delegate Transform(Delegate original)
        {
            var method = original.Method;
            var paramCount = method.GetParameters().Length;

            if (!typeof(Tensor).IsAssignableFrom(method.ReturnType))
                throw new NotSupportedException("pmap only supports functions returning Tensor");

            if (paramCount == 1 && method.GetParameters()[0].ParameterType == typeof(Tensor))
            {
                return CreateSingleInputWrapper((Func<Tensor, Tensor>)original);
            }

            if (paramCount == 2 &&
                method.GetParameters()[0].ParameterType == typeof(Tensor) &&
                method.GetParameters()[1].ParameterType == typeof(Tensor))
            {
                return CreateDoubleInputWrapper((Func<Tensor, Tensor, Tensor>)original);
            }

            throw new NotSupportedException("Unsupported delegate signature for pmap");
        }

        private Func<Tensor, Tensor> CreateSingleInputWrapper(Func<Tensor, Tensor> original)
        {
            int shardAxis = _shardAxes[0];
            int deviceCount = _mesh.DeviceCount;

            return (Tensor fullInput) =>
            {
                if (shardAxis == -1)
                {
                    // No sharding - execute on first device only
                    // For now, just execute locally
                    return original(fullInput);
                }

                // Shard input across devices
                var shardedInputs = ShardTensor(fullInput, shardAxis, deviceCount);
                var results = new List<Tensor>();

                // Execute on each device (for now, execute locally)
                // In reality, this would dispatch to actual devices
                for (int i = 0; i < deviceCount; i++)
                {
                    var shard = shardedInputs[i];
                    var result = original(shard);
                    results.Add(result);
                }

                // Gather results
                return GatherTensors(results, shardAxis);
            };
        }

        private Func<Tensor, Tensor, Tensor> CreateDoubleInputWrapper(Func<Tensor, Tensor, Tensor> original)
        {
            int shardAxis1 = _shardAxes[0];
            int shardAxis2 = _shardAxes[1];
            int deviceCount = _mesh.DeviceCount;

            return (Tensor input1, Tensor input2) =>
            {
                // Handle mixed sharding
                bool shardInput1 = shardAxis1 != -1;
                bool shardInput2 = shardAxis2 != -1;

                if (!shardInput1 && !shardInput2)
                {
                    return original(input1, input2);
                }

                var shardedInputs1 = shardInput1 ? ShardTensor(input1, shardAxis1, deviceCount) : null;
                var shardedInputs2 = shardInput2 ? ShardTensor(input2, shardAxis2, deviceCount) : null;
                var results = new List<Tensor>();

                for (int i = 0; i < deviceCount; i++)
                {
                    var shard1 = shardInput1 ? shardedInputs1[i] : input1;
                    var shard2 = shardInput2 ? shardedInputs2[i] : input2;
                    var result = original(shard1, shard2);
                    results.Add(result);
                }

                int outputAxis = shardInput1 ? shardAxis1 : shardAxis2;
                return GatherTensors(results, outputAxis);
            };
        }

        private Tensor[] ShardTensor(Tensor tensor, int axis, int shardCount)
        {
            if (tensor.Shape[axis] % shardCount != 0)
            {
                throw new ArgumentException($"Tensor dimension {axis} ({tensor.Shape[axis]}) must be divisible by shard count ({shardCount})");
            }

            int shardSize = tensor.Shape[axis] / shardCount;
            var shards = new List<Tensor>();

            for (int i = 0; i < shardCount; i++)
            {
                var start = i * shardSize;
                var shard = tensor.Slice(axis, start, start + shardSize);
                shards.Add(shard);
            }

            return shards.ToArray();
        }

        private Tensor GatherTensors(Tensor[] tensors, int axis)
        {
            return Tensor.Concat(tensors, axis);
        }
    }
}
```

### 3. Tensor Sharding Extensions

```csharp
public static class PMapTensorExtensions
{
    /// <summary>
    /// Slice tensor along axis with start and end indices.
    /// </summary>
    public static Tensor Slice(this Tensor tensor, int axis, int start, int end)
    {
        // Implementation: extract tensor[axis in range(start, end)]
        // This should use existing tensor slicing operations
        return tensor[new Slice[axis].From(start).To(end)];
    }

    /// <summary>
    /// Concatenate tensors along axis.
    /// </summary>
    public static Tensor Concat(Tensor[] tensors, int axis)
    {
        // Implementation: concatenate tensors along specified axis
        return Tensor.Concat(tensors, axis);
    }
}
```

## Files to Create
1. `src/MLFramework/Functional/Distributed/PMapTransform.cs`
2. `src/MLFramework/Functional/Distributed/PMapTensorExtensions.cs`
3. Update `src/MLFramework/Functional/Functional.cs` with Parallelize methods

## Dependencies
- spec_functional_core_interfaces.md
- spec_functional_device_mesh.md (must be completed first)
- MLFramework.Tensor with slice and concat operations

## Success Criteria
- Can parallelize single-input functions
- Can parallelize double-input functions
- Inputs are correctly sharded across devices
- Results are correctly gathered
- Works with 1D device meshes

## Notes for Coder
- This is a basic implementation - actual device dispatching will come later
- For now, execute all work locally (simulate parallelism)
- Focus on the sharding/gathering logic
- Device communication (all-reduce, etc.) will be in a future spec
- Ensure proper validation of sharding (divisibility check)

## Example Usage
```csharp
// Create device mesh
var mesh = DeviceMeshFactory.Create1D(4, DeviceType.GPU);

// Parallelize training step
void TrainStep(Tensor batch) { /* ... */ }
var parallelTrain = Functional.Parallelize(TrainStep, mesh);

// Now parallelTrain will shard the batch across 4 GPUs
```
