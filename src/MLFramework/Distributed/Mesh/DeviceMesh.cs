using MLFramework.Distributed;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Distributed.Mesh;

/// <summary>
/// Device mesh abstraction to enable flexible device assignment for 3D parallelism (TP + DP + PP).
/// The mesh organizes devices into a multi-dimensional grid, allowing different parallelism
/// dimensions to map to different subsets of devices.
/// </summary>
public class DeviceMesh
{
    private readonly int[] _shape;
    private readonly int _totalDevices;
    private readonly DeviceMeshCoord _myCoord;
    private readonly IProcessGroup _globalProcessGroup;
    private readonly Dictionary<(ParallelismDimension, int), List<int>> _processGroups;
    private readonly Dictionary<ParallelismDimension, int[]> _meshIndex;

    /// <summary>
    /// Gets the shape of the mesh (dimensions of each parallelism dimension).
    /// </summary>
    public IReadOnlyList<int> Shape => _shape;

    /// <summary>
    /// Gets this rank's coordinate in the mesh.
    /// </summary>
    public DeviceMeshCoord MyCoordinate => _myCoord;

    /// <summary>
    /// Gets the total number of devices in the mesh.
    /// </summary>
    public int TotalDevices => _totalDevices;

    /// <summary>
    /// Create a device mesh.
    /// </summary>
    /// <param name="shape">Dimensions of the mesh (e.g., [4, 2] for 4 DP groups x 2 TP ranks)</param>
    /// <param name="myCoord">This rank's coordinate in the mesh</param>
    /// <param name="globalProcessGroup">Global process group spanning all devices</param>
    public DeviceMesh(
        int[] shape,
        DeviceMeshCoord myCoord,
        IProcessGroup globalProcessGroup)
    {
        if (shape.Length != myCoord.Dimensions)
        {
            throw new ArgumentException(
                $"Shape dimensions ({shape.Length}) must match coordinate dimensions ({myCoord.Dimensions})");
        }

        _shape = shape;
        _myCoord = myCoord;
        _globalProcessGroup = globalProcessGroup ?? throw new ArgumentNullException(nameof(globalProcessGroup));
        _processGroups = new Dictionary<(ParallelismDimension, int), List<int>>();
        _meshIndex = new Dictionary<ParallelismDimension, int[]>();

        // Calculate total devices
        _totalDevices = 1;
        foreach (var dim in shape)
        {
            _totalDevices *= dim;
        }

        // Build mesh index for each dimension
        BuildMeshIndex();
    }

    /// <summary>
    /// Create a device mesh from a flat rank and shape.
    /// </summary>
    public static DeviceMesh CreateFromRank(int rank, int[] shape, IProcessGroup globalProcessGroup)
    {
        var coord = RankToCoord(rank, shape);
        return new DeviceMesh(shape, coord, globalProcessGroup);
    }

    /// <summary>
    /// Convert flat rank to mesh coordinates.
    /// </summary>
    private static DeviceMeshCoord RankToCoord(int rank, int[] shape)
    {
        var coords = new int[shape.Length];
        int remaining = rank;

        for (int i = shape.Length - 1; i >= 0; i--)
        {
            coords[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        return new DeviceMeshCoord(coords);
    }

    /// <summary>
    /// Convert mesh coordinates to flat rank.
    /// </summary>
    public int CoordToRank(DeviceMeshCoord coord)
    {
        if (coord.Dimensions != _shape.Length)
        {
            throw new ArgumentException("Coordinate dimensions must match mesh dimensions");
        }

        int rank = 0;
        int stride = 1;

        for (int i = _shape.Length - 1; i >= 0; i--)
        {
            rank += coord[i] * stride;
            stride *= _shape[i];
        }

        return rank;
    }

    /// <summary>
    /// Build mesh index for efficient group lookup.
    /// </summary>
    private void BuildMeshIndex()
    {
        foreach (ParallelismDimension dim in Enum.GetValues(typeof(ParallelismDimension)))
        {
            if ((int)dim >= _shape.Length)
                break;

            _meshIndex[dim] = new int[_shape[(int)dim]];
        }
    }

    /// <summary>
    /// Get information about a group for a specific dimension and index.
    /// For example: GetGroupRanks(ParallelismDimension.Tensor, 0) returns ranks in first TP group.
    /// </summary>
    public List<int> GetGroupRanks(ParallelismDimension dimension, int index = 0)
    {
        if ((int)dimension >= _shape.Length)
        {
            throw new ArgumentException(
                $"Dimension {dimension} not present in mesh (shape: {string.Join(",", _shape)})");
        }

        if (index < 0 || index >= _shape[(int)dimension])
        {
            throw new ArgumentException(
                $"Index {index} out of bounds for dimension {dimension} (size: {_shape[(int)dimension]})");
        }

        var key = (dimension, index);

        if (_processGroups.TryGetValue(key, out var ranks))
        {
            return ranks;
        }

        var ranksList = GetRanksInGroup(dimension, index);
        _processGroups[key] = ranksList;
        return ranksList;
    }

    /// <summary>
    /// Get all ranks that belong to a specific group.
    /// </summary>
    private List<int> GetRanksInGroup(ParallelismDimension dimension, int index)
    {
        var ranks = new List<int>();
        int dimIdx = (int)dimension;

        for (int i = 0; i < _totalDevices; i++)
        {
            var coord = RankToCoord(i, _shape);
            if (coord[dimIdx] == index)
            {
                ranks.Add(i);
            }
        }

        return ranks;
    }

    /// <summary>
    /// Get the TP group ranks for this rank's position in the DP dimension.
    /// </summary>
    public List<int> GetTPGroupRanks()
    {
        // TP group: fixed DP coordinate, varies TP coordinate
        // For 2D mesh [dp_size, tp_size]:
        // TP group = all ranks with same dp_idx
        int dpIdx = _myCoord[(int)ParallelismDimension.Data];
        return GetGroupRanks(ParallelismDimension.Data, dpIdx);
    }

    /// <summary>
    /// Get the DP group ranks for this rank's position in the TP dimension.
    /// </summary>
    public List<int> GetDPGroupRanks()
    {
        // DP group: fixed TP coordinate, varies DP coordinate
        // For 2D mesh [dp_size, tp_size]:
        // DP group = all ranks with same tp_idx
        int tpIdx = _myCoord[(int)ParallelismDimension.Tensor];
        return GetGroupRanks(ParallelismDimension.Tensor, tpIdx);
    }

    /// <summary>
    /// Get the global process group.
    /// </summary>
    public IProcessGroup GetGlobalProcessGroup()
    {
        return _globalProcessGroup;
    }

    /// <summary>
    /// Barrier across all devices in the mesh.
    /// </summary>
    public async Task BarrierAsync()
    {
        await _globalProcessGroup.BarrierAsync();
    }

    /// <summary>
    /// Barrier across a specific dimension.
    /// Note: In a real implementation, this would use sub-process groups.
    /// For now, this falls back to the global barrier.
    /// </summary>
    public async Task BarrierAsync(ParallelismDimension dimension)
    {
        // TODO: Implement sub-process group barriers
        // For now, use global barrier
        await _globalProcessGroup.BarrierAsync();
    }
}
