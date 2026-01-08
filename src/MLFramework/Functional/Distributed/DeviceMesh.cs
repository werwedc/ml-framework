using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Functional.Distributed
{
    /// <summary>
    /// A mesh of devices for distributed computation.
    /// Supports N-dimensional topologies for data and model parallelism.
    /// </summary>
    public class DeviceMesh
    {
        private readonly Device[] _devices;
        private readonly int[] _shape;
        private readonly Dictionary<string, Device> _deviceMap;

        /// <summary>
        /// Shape of the mesh (e.g., [8] for 1D, [2, 4] for 2D).
        /// </summary>
        public IReadOnlyList<int> Shape => _shape;

        /// <summary>
        /// Total number of devices in the mesh.
        /// </summary>
        public int DeviceCount => _devices.Length;

        /// <summary>
        /// Number of dimensions in the mesh.
        /// </summary>
        public int Rank => _shape.Length;

        /// <summary>
        /// Create a 1D device mesh.
        /// </summary>
        public DeviceMesh(Device[] devices)
        {
            if (devices == null || devices.Length == 0)
                throw new ArgumentException("Must provide at least one device");

            _devices = devices.ToArray();
            _shape = new int[] { _devices.Length };
            _deviceMap = new Dictionary<string, Device>();

            for (int i = 0; i < _devices.Length; i++)
            {
                _deviceMap[_devices[i].Name] = _devices[i];
            }
        }

        /// <summary>
        /// Create an N-dimensional device mesh.
        /// </summary>
        /// <param name="shape">Shape of the mesh (e.g., [2, 4] for 2D mesh with 8 devices)</param>
        /// <param name="devices">List of devices to use</param>
        public DeviceMesh(int[] shape, Device[] devices)
        {
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension");

            int totalDevices = shape.Aggregate(1, (a, b) => a * b);
            if (devices.Length != totalDevices)
            {
                throw new ArgumentException($"Device count ({devices.Length}) must match mesh size ({totalDevices})");
            }

            _shape = shape.ToArray();
            _devices = devices.ToArray();
            _deviceMap = new Dictionary<string, Device>();

            for (int i = 0; i < _devices.Length; i++)
            {
                _deviceMap[_devices[i].Name] = _devices[i];
            }
        }

        /// <summary>
        /// Get device at specific mesh coordinates.
        /// </summary>
        public Device GetDevice(params int[] indices)
        {
            if (indices.Length != Rank)
                throw new ArgumentException($"Expected {Rank} indices, got {indices.Length}");

            int flatIndex = ComputeFlatIndex(indices);
            return _devices[flatIndex];
        }

        /// <summary>
        /// Get all devices along a specific axis.
        /// </summary>
        public Device[] GetAxisDevices(int axis)
        {
            if (axis < 0 || axis >= Rank)
                throw new ArgumentException($"Invalid axis {axis} for mesh of rank {Rank}");

            return _devices.Distinct().ToArray();  // Simplified - in reality, group by axis
        }

        /// <summary>
        /// Sharding axis for distributing data.
        /// </summary>
        public class ShardingAxis
        {
            public string Name { get; }
            public int AxisIndex { get; }

            public ShardingAxis(string name, int axisIndex)
            {
                Name = name;
                AxisIndex = axisIndex;
            }
        }

        /// <summary>
        /// Get a sharding axis by name.
        /// </summary>
        public ShardingAxis this[string name]
        {
            get
            {
                if (name == "data" || name == "batch")
                {
                    return new ShardingAxis(name, 0);  // Default to first axis
                }
                throw new ArgumentException($"Unknown axis: {name}");
            }
        }

        private int ComputeFlatIndex(int[] indices)
        {
            // Convert multi-dimensional index to flat array index (row-major order)
            int index = 0;
            int stride = 1;

            for (int i = Rank - 1; i >= 0; i--)
            {
                if (indices[i] < 0 || indices[i] >= _shape[i])
                    throw new IndexOutOfRangeException($"Index {indices[i]} out of range for axis {i}");

                index += indices[i] * stride;
                stride *= _shape[i];
            }

            return index;
        }
    }
}
