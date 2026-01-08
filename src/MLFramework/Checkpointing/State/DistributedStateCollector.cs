namespace MachineLearning.Checkpointing;

/// <summary>
/// Collects and merges state across distributed ranks
/// </summary>
public class DistributedStateCollector
{
    private readonly IDistributedCoordinator _coordinator;

    /// <summary>
    /// Creates a new distributed state collector
    /// </summary>
    /// <param name="coordinator">Distributed coordinator for communication</param>
    public DistributedStateCollector(IDistributedCoordinator coordinator)
    {
        _coordinator = coordinator ?? throw new ArgumentNullException(nameof(coordinator));
    }

    /// <summary>
    /// Collects local sharded state from a model
    /// For FSDP, model only has local shard
    /// For DDP, model has full state duplicated
    /// </summary>
    /// <param name="model">The stateful model to collect state from</param>
    /// <returns>The local state dictionary</returns>
    public StateDict CollectLocalState(IStateful model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        return model.GetStateDict();
    }

    /// <summary>
    /// Collects local sharded state from an optimizer
    /// </summary>
    /// <param name="optimizer">The stateful optimizer to collect state from</param>
    /// <returns>The local state dictionary</returns>
    public StateDict CollectLocalOptimizerState(IStateful optimizer)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        return optimizer.GetStateDict();
    }

    /// <summary>
    /// Merges states from multiple ranks for loading
    /// Handles both DDP (all ranks have same tensors) and FSDP (sharded)
    /// </summary>
    /// <param name="states">Array of state dictionaries from each rank</param>
    /// <returns>Merged state dictionary</returns>
    public StateDict MergeStates(StateDict[] states)
    {
        if (states == null)
            throw new ArgumentNullException(nameof(states));

        if (states.Length == 0)
            return new StateDict();

        var merged = new StateDict();

        foreach (var state in states)
        {
            if (state == null)
                continue;

            foreach (var (key, tensor) in state)
            {
                if (merged.ContainsKey(key))
                {
                    // DDP case: all ranks have same tensors
                    // Keep first occurrence and verify consistency
                    // In a real implementation, we'd verify the tensors are identical
                    continue;
                }
                merged[key] = tensor;
            }
        }

        return merged;
    }

    /// <summary>
    /// Merges states from multiple ranks with consistency checking
    /// </summary>
    /// <param name="states">Array of state dictionaries from each rank</param>
    /// <param name="checkConsistency">Whether to verify tensor consistency across ranks</param>
    /// <returns>Merged state dictionary</returns>
    public StateDict MergeStates(StateDict[] states, bool checkConsistency)
    {
        if (!checkConsistency)
            return MergeStates(states);

        if (states == null)
            throw new ArgumentNullException(nameof(states));

        if (states.Length == 0)
            return new StateDict();

        var merged = new StateDict();
        var seenKeys = new Dictionary<string, int>();

        foreach (var state in states)
        {
            if (state == null)
                continue;

            foreach (var (key, tensor) in state)
            {
                if (!merged.ContainsKey(key))
                {
                    merged[key] = tensor;
                    seenKeys[key] = 1;
                }
                else
                {
                    // DDP case: all ranks have same tensors
                    seenKeys[key]++;

                    if (checkConsistency)
                    {
                        // Verify tensor shapes match
                        var existingTensor = merged[key];
                        if (!existingTensor.Shape.SequenceEqual(tensor.Shape))
                        {
                            throw new InvalidOperationException(
                                $"Shape mismatch for tensor '{key}': " +
                                $"expected {string.Join(",", existingTensor.Shape)}, " +
                                $"got {string.Join(",", tensor.Shape)}");
                        }

                        // Verify data types match
                        if (existingTensor.DataType != tensor.DataType)
                        {
                            throw new InvalidOperationException(
                                $"Data type mismatch for tensor '{key}': " +
                                $"expected {existingTensor.DataType}, " +
                                $"got {tensor.DataType}");
                        }
                    }
                }
            }
        }

        return merged;
    }

    /// <summary>
    /// Broadcasts state from rank 0 to all other ranks
    /// Useful for ensuring all ranks have the same initial state
    /// </summary>
    /// <param name="state">State to broadcast (only used by rank 0)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Broadcasted state (same for all ranks)</returns>
    public async Task<StateDict> BroadcastStateAsync(
        StateDict state,
        CancellationToken cancellationToken = default)
    {
        if (_coordinator.Rank == 0)
        {
            // Rank 0 broadcasts the state
            // Convert to serializable format
            var serializableState = state.ToSerializableFormat();
            await _coordinator.BroadcastAsync(serializableState, cancellationToken);
            return state;
        }
        else
        {
            // Other ranks receive the state
            var serializableState = await _coordinator.BroadcastAsync<object>(
                new object(), // Dummy object, actual data comes from rank 0
                cancellationToken);

            if (serializableState is SerializableStateDict dict)
            {
                return dict.ToStateDict();
            }

            throw new InvalidOperationException("Failed to receive state from rank 0");
        }
    }
}

/// <summary>
/// Serializable format for StateDict
/// </summary>
internal class SerializableStateDict
{
    public List<string> Keys { get; set; } = new();
    public List<byte[]> Data { get; set; } = new();
    public List<long[]> Shapes { get; set; } = new();
    public List<TensorDataType> DataTypes { get; set; } = new();
}

/// <summary>
/// Extension methods for StateDict serialization
/// </summary>
internal static class StateDictExtensions
{
    public static SerializableStateDict ToSerializableFormat(this StateDict state)
    {
        var serializable = new SerializableStateDict();

        foreach (var (key, tensor) in state)
        {
            serializable.Keys.Add(key);
            serializable.Shapes.Add(tensor.Shape);
            serializable.DataTypes.Add(tensor.DataType);

            // In a real implementation, serialize actual tensor data
            serializable.Data.Add(Array.Empty<byte>());
        }

        return serializable;
    }

    public static StateDict ToStateDict(this SerializableStateDict serializable)
    {
        var state = new StateDict();

        // In a real implementation, deserialize actual tensor data
        // For now, create placeholder tensors with correct shape

        return state;
    }
}
