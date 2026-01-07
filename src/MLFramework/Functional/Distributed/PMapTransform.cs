using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Functional.Distributed
{
    /// <summary>
    /// Implements the pmap (parallel map) transformation for SPMD (Single Program Multiple Data)
    /// execution across a device mesh.
    /// </summary>
    public class PMapTransform : BaseTransformation
    {
        private readonly DeviceMesh _mesh;
        private readonly int[] _shardAxes;  // -1 means don't shard

        /// <summary>
        /// Initializes a new instance of the <see cref="PMapTransform"/> class.
        /// </summary>
        /// <param name="original">The original function to parallelize.</param>
        /// <param name="mesh">Device mesh for distribution.</param>
        /// <param name="in_axes">Which axes to shard across devices (default: data axis).</param>
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

        /// <summary>
        /// Applies the pmap transformation to a delegate.
        /// </summary>
        /// <param name="original">The original delegate to transform.</param>
        /// <returns>A new delegate with the pmap transformation applied.</returns>
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

        /// <summary>
        /// Creates a wrapper function for single-input delegates.
        /// </summary>
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
                return GatherTensors(results.ToArray(), shardAxis);
            };
        }

        /// <summary>
        /// Creates a wrapper function for double-input delegates.
        /// </summary>
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
                return GatherTensors(results.ToArray(), outputAxis);
            };
        }

        /// <summary>
        /// Shards a tensor along the specified axis.
        /// </summary>
        private Tensor[] ShardTensor(Tensor tensor, int axis, int shardCount)
        {
            if (axis >= tensor.Dimensions)
            {
                throw new ArgumentException(
                    $"Axis {axis} is out of bounds for tensor with {tensor.Dimensions} dimensions");
            }

            if (tensor.Shape[axis] % shardCount != 0)
            {
                throw new ArgumentException(
                    $"Tensor dimension {axis} ({tensor.Shape[axis]}) must be divisible by shard count ({shardCount})");
            }

            int shardSize = tensor.Shape[axis] / shardCount;
            var shards = new Tensor[shardCount];

            for (int i = 0; i < shardCount; i++)
            {
                var start = i * shardSize;
                var end = start + shardSize;
                shards[i] = PMapTensorExtensions.Slice(tensor, axis, start, end);
            }

            return shards;
        }

        /// <summary>
        /// Gathers tensors along the specified axis.
        /// </summary>
        private Tensor GatherTensors(Tensor[] tensors, int axis)
        {
            return PMapTensorExtensions.Concat(tensors, axis);
        }
    }
}
