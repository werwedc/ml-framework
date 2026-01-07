# Spec: Device Mesh Support for 3D Parallelism

## Overview
Implement device mesh abstraction to enable flexible device assignment for 3D parallelism (TP + DP + PP). The mesh organizes devices into a multi-dimensional grid, allowing different parallelism dimensions to map to different subsets of devices.

## Context
For large-scale training, we combine:
- **Data Parallelism (DP)**: Replicate model across multiple groups
- **Tensor Parallelism (TP)**: Split tensors within each DP group
- **Pipeline Parallelism (PP)**: Split layers across devices (future)

A device mesh provides a clean way to organize and communicate across these dimensions.

## Implementation Details

### 1. Device Mesh Coordinates

```csharp
namespace MLFramework.Distributed.Mesh;

public struct DeviceMeshCoord
{
    public readonly int[] Coordinates;

    public DeviceMeshCoord(params int[] coords)
    {
        Coordinates = coords;
    }

    public int this[int dim]
    {
        get => Coordinates[dim];
        set => Coordinates[dim] = value;
    }

    public int Dimensions => Coordinates.Length;

    public override string ToString()
    {
        return $"({string.Join(", ", Coordinates)})";
    }

    public bool Equals(DeviceMeshCoord other)
    {
        if (Dimensions != other.Dimensions)
            return false;
        for (int i = 0; i < Dimensions; i++)
        {
            if (Coordinates[i] != other.Coordinates[i])
                return false;
        }
        return true;
    }
}
```

### 2. Device Mesh Topology

```csharp
public enum ParallelismDimension
{
    Data = 0,      // DP: split across batches
    Tensor = 1,     // TP: split within layers
    Pipeline = 2,   // PP: split across layers (future)
    Expert = 3      // MoE: split across experts (future)
}

public class DeviceMesh
{
    private readonly int[] _shape;
    private readonly int _totalDevices;
    private readonly DeviceMeshCoord _myCoord;
    private readonly ICommunicator _globalCommunicator;
    private readonly Dictionary<(ParallelismDimension, int), ProcessGroup> _processGroups;
    private readonly Dictionary<ParallelismDimension, int[]> _meshIndex;

    public IReadOnlyList<int> Shape => _shape;
    public DeviceMeshCoord MyCoordinate => _myCoord;
    public int TotalDevices => _totalDevices;

    /// <summary>
    /// Create a device mesh
    /// shape: dimensions of the mesh (e.g., [4, 2] for 4 DP groups x 2 TP ranks)
    /// myCoord: this rank's coordinate in the mesh
    /// globalComm: global communicator spanning all devices
    /// </summary>
    public DeviceMesh(
        int[] shape,
        DeviceMeshCoord myCoord,
        ICommunicator globalComm)
    {
        if (shape.Length != myCoord.Dimensions)
        {
            throw new ArgumentException(
                $"Shape dimensions ({shape.Length}) must match coordinate dimensions ({myCoord.Dimensions})");
        }

        _shape = shape;
        _myCoord = myCoord;
        _globalCommunicator = globalComm;
        _processGroups = new Dictionary<(ParallelismDimension, int), ProcessGroup>();
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
    /// Create a device mesh from a flat rank and shape
    /// </summary>
    public static DeviceMesh CreateFromRank(int rank, int[] shape, ICommunicator globalComm)
    {
        var coord = RankToCoord(rank, shape);
        return new DeviceMesh(shape, coord, globalComm);
    }

    /// <summary>
    /// Convert flat rank to mesh coordinates
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
    /// Convert mesh coordinates to flat rank
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
    /// Build mesh index for efficient group lookup
    /// </summary>
    private void BuildMeshIndex()
    {
        foreach (ParallelismDimension dim in Enum.GetValues<ParallelismDimension>())
        {
            if ((int)dim >= _shape.Length)
                break;

            _meshIndex[dim] = new int[_shape[(int)dim]];
        }
    }

    /// <summary>
    /// Get or create a process group for a specific dimension and index
    /// For example: GetGroup(ParallelismDimension.Tensor, 0) returns first TP group
    /// </summary>
    public ProcessGroup GetGroup(ParallelismDimension dimension, int index = 0)
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

        if (_processGroups.TryGetValue(key, out var group))
        {
            return group;
        }

        // Create process group
        var ranks = GetRanksInGroup(dimension, index);
        group = new ProcessGroup(_globalCommunicator, ranks, CoordToRank(_myCoord));
        _processGroups[key] = group;
        return group;
    }

    /// <summary>
    /// Get all ranks that belong to a specific group
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
    /// Get the TP group for this rank's position in the DP dimension
    /// </summary>
    public ProcessGroup GetTPGroup()
    {
        // TP group: fixed DP coordinate, varies TP coordinate
        // For 2D mesh [dp_size, tp_size]:
        // TP group = all ranks with same dp_idx
        int dpIdx = _myCoord[(int)ParallelismDimension.Data];
        return GetGroup(ParallelismDimension.Data, dpIdx);
    }

    /// <summary>
    /// Get the DP group for this rank's position in the TP dimension
    /// </summary>
    public ProcessGroup GetDPGroup()
    {
        // DP group: fixed TP coordinate, varies DP coordinate
        // For 2D mesh [dp_size, tp_size]:
        // DP group = all ranks with same tp_idx
        int tpIdx = _myCoord[(int)ParallelismDimension.Tensor];
        return GetGroup(ParallelismDimension.Tensor, tpIdx);
    }

    /// <summary>
    /// Get the global group (all devices)
    /// </summary>
    public ProcessGroup GetGlobalGroup()
    {
        return new ProcessGroup(
            _globalCommunicator,
            Enumerable.Range(0, _totalDevices).ToList(),
            CoordToRank(_myCoord));
    }

    /// <summary>
    /// Barrier across all devices in the mesh
    /// </summary>
    public async Task BarrierAsync()
    {
        await _globalCommunicator.BarrierAsync();
    }

    /// <summary>
    /// Barrier across a specific dimension
    /// </summary>
    public async Task BarrierAsync(ParallelismDimension dimension)
    {
        var group = GetGroup(dimension, _myCoord[(int)dimension]);
        await group.BarrierAsync();
    }
}
```

### 3. Mesh-aware Context Manager

```csharp
public class TensorParallelMeshContext : TensorParallelContext
{
    private readonly DeviceMesh _mesh;
    private readonly ProcessGroup _tpGroup;

    public DeviceMesh Mesh => _mesh;
    public ProcessGroup TPGroup => _tpGroup;

    public TensorParallelMeshContext(
        DeviceMesh mesh,
        ICommunicator globalComm,
        bool ownsCommunicator = true)
        : base(globalComm, ownsCommunicator)
    {
        _mesh = mesh;
        _tpGroup = mesh.GetTPGroup();
    }

    /// <summary>
    /// Initialize TP context with device mesh
    /// </summary>
    public static TensorParallelMeshContext InitializeWithMesh(
        int[] meshShape,
        int rank,
        string backend = "mock")
    {
        var config = new Dictionary<string, object>
        {
            ["world_size"] = meshShape.Aggregate(1, (a, b) => a * b),
            ["rank"] = rank
        };
        var globalComm = CommunicatorFactory.Create(backend, config);
        var mesh = DeviceMesh.CreateFromRank(rank, meshShape, globalComm);

        return new TensorParallelMeshContext(mesh, globalComm, ownsCommunicator: true);
    }

    /// <summary>
    /// Get the TP process group for this rank
    /// </summary>
    public ProcessGroup GetTPProcessGroup()
    {
        return _tpGroup;
    }

    /// <summary>
    /// Get the DP process group for this rank (for hybrid TP+DP)
    /// </summary>
    public ProcessGroup GetDPProcessGroup()
    {
        return _mesh.GetDPGroup();
    }
}
```

### 4. Mesh-aware Layer Factory

```csharp
public static class TPMeshLayerFactory
{
    /// <summary>
    /// Create TP layer with mesh-aware process group
    /// </summary>
    public static ColumnParallelLinear CreateColumnParallel(
        DeviceMesh mesh,
        int inputSize,
        int outputSize,
        bool bias = true,
        bool gatherOutput = false)
    {
        var tpGroup = mesh.GetTPGroup();
        return new ColumnParallelLinear(
            inputSize, outputSize, bias, gatherOutput, tpGroup);
    }

    /// <summary>
    /// Create row parallel linear with mesh-aware process group
    /// </summary>
    public static RowParallelLinear CreateRowParallel(
        DeviceMesh mesh,
        int inputSize,
        int outputSize,
        bool bias = true,
        bool inputIsSharded = true)
    {
        var tpGroup = mesh.GetTPGroup();
        return new RowParallelLinear(
            inputSize, outputSize, bias, inputIsSharded, tpGroup);
    }

    /// <summary>
    /// Create an MLP block that is mesh-aware
    /// </summary>
    public static (ColumnParallelLinear, RowParallelLinear) CreateMLPBlock(
        DeviceMesh mesh,
        int inputSize,
        int hiddenSize,
        int outputSize,
        bool bias = true)
    {
        var column = CreateColumnParallel(mesh, inputSize, hiddenSize, bias, gatherOutput: false);
        var row = CreateRowParallel(mesh, hiddenSize, outputSize, bias, inputIsSharded: true);
        return (column, row);
    }
}
```

### 5. Mesh State Helpers

```csharp
public static class MeshState
{
    /// <summary>
    /// Get the TP world size from mesh
    /// </summary>
    public static int GetTPWorldSize(DeviceMesh mesh)
    {
        return mesh.Shape[(int)ParallelismDimension.Tensor];
    }

    /// <summary>
    /// Get the DP world size from mesh
    /// </summary>
    public static int GetDPWorldSize(DeviceMesh mesh)
    {
        return mesh.Shape[(int)ParallelismDimension.Data];
    }

    /// <summary>
    /// Check if this is the master rank (0, 0, ...)
    /// </summary>
    public static bool IsMasterRank(DeviceMesh mesh)
    {
        foreach (var coord in mesh.MyCoordinate.Coordinates)
        {
            if (coord != 0)
                return false;
        }
        return true;
    }

    /// <summary>
    /// Execute code only on master rank (0, 0, ...)
    /// </summary>
    public static Task ExecuteOnMasterAsync(DeviceMesh mesh, Func<Task> action)
    {
        if (IsMasterRank(mesh))
        {
            return action();
        }
        return Task.CompletedTask;
    }
}
```

## Files to Create

### Source Files
- `src/MLFramework/Distributed/Mesh/DeviceMeshCoord.cs`
- `src/MLFramework/Distributed/Mesh/DeviceMesh.cs`
- `src/MLFramework/Distributed/Mesh/ParallelismDimension.cs`
- `src/MLFramework/Distributed/Mesh/TensorParallelMeshContext.cs`
- `src/MLFramework/Distributed/Mesh/TPMeshLayerFactory.cs`
- `src/MLFramework/Distributed/Mesh/MeshState.cs`

### Test Files
- `tests/MLFramework.Tests/Distributed/Mesh/DeviceMeshTests.cs`
- `tests/MLFramework.Tests/Distributed/Mesh/TPMeshContextTests.cs`

## Test Requirements

1. **Device Mesh Tests**
   - Test mesh creation with various shapes
   - Test RankToCoord and CoordToRank conversions
   - Test shape dimensions match

2. **Process Group Tests**
   - Test GetGroup() creates correct process groups
   - Test GetTPGroup() returns correct TP group
   - Test GetDPGroup() returns correct DP group
   - Verify group members are correct

3. **Barrier Tests**
   - Test global barrier across all devices
   - Test dimension-specific barriers

4. **Mesh Context Tests**
   - Test InitializeWithMesh creates valid context
   - Test TP and DP process groups are correct
   - Test mesh shape is accessible

5. **Layer Factory Tests**
   - Test layers are created with correct process groups
   - Test layers use mesh groups for communication

6. **Edge Cases**
   - Test with 1D mesh (single dimension)
   - Test with 3D mesh (DP + TP + PP)
   - Test with non-rectangular shapes

## Dependencies
- `ICommunicator` and `ProcessGroup` from communication primitives
- `TensorParallelContext` from TP context manager
- TP layers from previous specs

## Success Criteria
- [ ] DeviceMesh correctly maps ranks to coordinates
- [ ] Process groups are correctly created for each dimension
- [ ] TP and DP groups are correctly identified
- [ ] Barriers work across specific dimensions
- [ ] Mesh context properly integrates with TP
- [ ] Layers created with mesh use correct process groups
- [ ] Unit tests pass for all scenarios

## Estimated Time
45-60 minutes
