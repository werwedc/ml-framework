namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Mock implementation of ITensor for testing
/// </summary>
public class MockTensor : ITensor
{
    public long[] Shape { get; set; } = Array.Empty<long>();
    public TensorDataType DataType { get; set; } = TensorDataType.Float32;
    private readonly long _size;

    public MockTensor(long[] shape, TensorDataType dataType, long size)
    {
        Shape = shape;
        DataType = dataType;
        _size = size;
    }

    public long GetSizeInBytes()
    {
        return _size;
    }
}

/// <summary>
/// Mock implementation of IDistributedModel for testing
/// </summary>
public class MockDistributedModel : IDistributedModel
{
    public DistributedStrategy Strategy { get; set; }
    public int WorldSize { get; set; }
    public int Rank { get; set; }
    private readonly StateDict _stateDict;

    public MockDistributedModel(
        DistributedStrategy strategy,
        int worldSize,
        int rank,
        StateDict? stateDict = null)
    {
        Strategy = strategy;
        WorldSize = worldSize;
        Rank = rank;
        _stateDict = stateDict ?? new StateDict();
    }

    public StateDict GetStateDict()
    {
        return _stateDict;
    }

    public void LoadStateDict(StateDict state)
    {
        foreach (var (key, value) in state)
        {
            _stateDict[key] = value;
        }
    }

    public StateDict GetLocalStateDict()
    {
        return _stateDict;
    }

    public StateDict GetFullStateDict()
    {
        return _stateDict;
    }

    public void LoadLocalStateDict(StateDict state)
    {
        LoadStateDict(state);
    }

    public void LoadFullStateDict(StateDict state)
    {
        LoadStateDict(state);
    }
}

/// <summary>
/// Mock implementation of IDistributedCoordinator for testing
/// </summary>
public class MockDistributedCoordinator : IDistributedCoordinator
{
    public int WorldSize { get; set; }
    public int Rank { get; set; }

    public MockDistributedCoordinator(int worldSize, int rank)
    {
        WorldSize = worldSize;
        Rank = rank;
    }
}

/// <summary>
/// Mock implementation of IStateful for testing
/// </summary>
public class MockStateful : IStateful
{
    private readonly StateDict _stateDict;

    public MockStateful(StateDict? stateDict = null)
    {
        _stateDict = stateDict ?? new StateDict();
    }

    public StateDict GetStateDict()
    {
        return _stateDict;
    }

    public void LoadStateDict(StateDict state)
    {
        foreach (var (key, value) in state)
        {
            _stateDict[key] = value;
        }
    }
}
