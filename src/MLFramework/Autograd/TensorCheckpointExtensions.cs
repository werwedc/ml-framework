using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Extension methods for Tensor to support gradient checkpointing.
/// </summary>
public static class TensorCheckpointExtensions
{
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, bool> _checkpointedTensors = new();

    /// <summary>
    /// Marks this tensor as a checkpoint for gradient computation.
    /// The tensor's activations will be saved for recompute during backward pass.
    /// </summary>
    /// <param name="tensor">The tensor to mark as a checkpoint.</param>
    /// <returns>The same tensor for method chaining.</returns>
    public static Tensor MarkCheckpoint(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var tensorId = tensor.GetHashCode();
        _checkpointedTensors.TryAdd(tensorId, true);

        return tensor;
    }

    /// <summary>
    /// Checks whether this tensor is marked as a checkpoint.
    /// </summary>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if the tensor is a checkpoint, false otherwise.</returns>
    public static bool IsCheckpoint(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var tensorId = tensor.GetHashCode();
        return _checkpointedTensors.ContainsKey(tensorId);
    }

    /// <summary>
    /// Removes the checkpoint mark from this tensor.
    /// </summary>
    /// <param name="tensor">The tensor to unmark.</param>
    /// <returns>The same tensor for method chaining.</returns>
    public static Tensor UnmarkCheckpoint(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var tensorId = tensor.GetHashCode();
        _checkpointedTensors.TryRemove(tensorId, out _);

        return tensor;
    }

    /// <summary>
    /// Marks this tensor as a checkpoint with the specified checkpoint scope.
    /// </summary>
    /// <param name="tensor">The tensor to mark as a checkpoint.</param>
    /// <param name="scopeName">The name of the checkpoint scope to associate with.</param>
    /// <returns>The same tensor for method chaining.</returns>
    public static Tensor MarkCheckpoint(this Tensor tensor, string scopeName)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (string.IsNullOrEmpty(scopeName))
            throw new ArgumentException("Scope name cannot be null or empty", nameof(scopeName));

        var scope = CheckpointScope.GetActiveScope(scopeName);
        if (scope == null)
            throw new InvalidOperationException($"No active checkpoint scope found with name '{scopeName}'");

        if (!scope.IsEnabled)
            return tensor;

        // Mark the tensor as checkpointed
        MarkCheckpoint(tensor);

        return tensor;
    }

    /// <summary>
    /// Marks this tensor as a checkpoint with the specified checkpoint scope.
    /// </summary>
    /// <param name="tensor">The tensor to mark as a checkpoint.</param>
    /// <param name="scope">The checkpoint scope to associate with.</param>
    /// <returns>The same tensor for method chaining.</returns>
    public static Tensor MarkCheckpoint(this Tensor tensor, CheckpointScope scope)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (scope == null)
            throw new ArgumentNullException(nameof(scope));

        if (!scope.IsEnabled)
            return tensor;

        // Mark the tensor as checkpointed
        MarkCheckpoint(tensor);

        return tensor;
    }

    /// <summary>
    /// Clears all checkpoint marks from all tensors.
    /// </summary>
    public static void ClearAllCheckpointMarks()
    {
        _checkpointedTensors.Clear();
    }

    /// <summary>
    /// Gets the count of tensors marked as checkpoints.
    /// </summary>
    /// <returns>The count of checkpointed tensors.</returns>
    public static int GetCheckpointCount()
    {
        return _checkpointedTensors.Count;
    }

    /// <summary>
    /// Checks if a tensor requires recomputation during backward pass.
    /// This is true if the tensor is marked as a checkpoint and has a recompute function.
    /// </summary>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if the tensor requires recomputation, false otherwise.</returns>
    public static bool RequiresRecomputation(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return IsCheckpoint(tensor) && tensor.RequiresGrad;
    }

    /// <summary>
    /// Creates a checkpoint node for this tensor with the specified scope.
    /// </summary>
    /// <param name="tensor">The tensor to create a checkpoint node for.</param>
    /// <param name="operation">The operation context.</param>
    /// <param name="scope">The checkpoint scope.</param>
    /// <param name="children">The child nodes.</param>
    /// <returns>A new checkpoint node for this tensor.</returns>
    public static CheckpointNode CreateCheckpointNode(
        this Tensor tensor,
        OperationContext operation,
        CheckpointScope scope,
        params GraphNode[] children)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        if (scope == null)
            throw new ArgumentNullException(nameof(scope));

        var node = new CheckpointNode(tensor, operation, scope, children);
        tensor.MarkCheckpoint(scope);
        return node;
    }

    /// <summary>
    /// Creates a checkpoint node for this tensor without a scope.
    /// </summary>
    /// <param name="tensor">The tensor to create a checkpoint node for.</param>
    /// <param name="operation">The operation context.</param>
    /// <param name="children">The child nodes.</param>
    /// <returns>A new checkpoint node for this tensor.</returns>
    public static CheckpointNode CreateCheckpointNode(
        this Tensor tensor,
        OperationContext operation,
        params GraphNode[] children)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        var node = new CheckpointNode(tensor, operation, children);
        tensor.MarkCheckpoint();
        return node;
    }
}
