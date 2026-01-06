using RitterFramework.Core.Tensor;
using System.Collections.Concurrent;

namespace MLFramework.Autograd;

/// <summary>
/// Represents a node in the computational graph that can be checkpointed.
/// Checkpoint nodes save intermediate activations selectively to reduce memory usage.
/// </summary>
public class CheckpointNode : GraphNode
{
    private readonly object _lock = new object();
    private List<Tensor> _savedActivations;
    private List<Func<Tensor[]>> _recomputeFunctions;
    private bool _recomputed;

    /// <summary>
    /// Gets or sets whether this node is marked as a checkpoint.
    /// </summary>
    public bool IsCheckpoint { get; set; }

    /// <summary>
    /// Gets the list of saved activations at this checkpoint.
    /// </summary>
    public List<Tensor> SavedActivations => _savedActivations;

    /// <summary>
    /// Gets the list of recompute functions for this node.
    /// </summary>
    public List<Func<Tensor[]>> RecomputeFunctions => _recomputeFunctions;

    /// <summary>
    /// Gets the checkpoint scope associated with this node.
    /// </summary>
    public CheckpointScope? Scope { get; private set; }

    /// <summary>
    /// Gets whether this node has been recomputed during backward pass.
    /// </summary>
    public bool HasRecomputed => _recomputed;

    /// <summary>
    /// Initializes a new instance of the CheckpointNode class.
    /// </summary>
    /// <param name="output">The output tensor produced by this operation.</param>
    /// <param name="operation">The operation context for this node.</param>
    /// <param name="children">The child nodes (inputs) that this node depends on.</param>
    public CheckpointNode(Tensor output, OperationContext operation, params GraphNode[] children)
        : base(output, operation, children)
    {
        _savedActivations = new List<Tensor>();
        _recomputeFunctions = new List<Func<Tensor[]>>();
        _recomputed = false;
        IsCheckpoint = false;
    }

    /// <summary>
    /// Initializes a new instance of the CheckpointNode class with a checkpoint scope.
    /// </summary>
    /// <param name="output">The output tensor produced by this operation.</param>
    /// <param name="operation">The operation context for this node.</param>
    /// <param name="scope">The checkpoint scope associated with this node.</param>
    /// <param name="children">The child nodes (inputs) that this node depends on.</param>
    public CheckpointNode(Tensor output, OperationContext operation, CheckpointScope scope, params GraphNode[] children)
        : this(output, operation, children)
    {
        Scope = scope ?? throw new ArgumentNullException(nameof(scope));
        IsCheckpoint = scope.IsEnabled;
        scope.RegisterNode(this);
    }

    /// <summary>
    /// Saves activations at this checkpoint.
    /// </summary>
    /// <param name="activations">The activations to save.</param>
    public void SaveActivations(params Tensor[] activations)
    {
        if (activations == null)
            throw new ArgumentNullException(nameof(activations));

        lock (_lock)
        {
            foreach (var activation in activations)
            {
                if (activation != null)
                {
                    _savedActivations.Add(activation.Clone());
                }
            }
        }
    }

    /// <summary>
    /// Adds a recompute function that can recreate the activations.
    /// </summary>
    /// <param name="recomputeFn">The function to recompute activations.</param>
    public void AddRecomputeFunction(Func<Tensor[]> recomputeFn)
    {
        if (recomputeFn == null)
            throw new ArgumentNullException(nameof(recomputeFn));

        lock (_lock)
        {
            _recomputeFunctions.Add(recomputeFn);
        }
    }

    /// <summary>
    /// Recomputes the activations from stored recompute functions.
    /// </summary>
    /// <returns>The recomputed activations.</returns>
    public Tensor[] Recompute()
    {
        lock (_lock)
        {
            if (_recomputeFunctions.Count == 0)
                throw new InvalidOperationException("No recompute functions available");

            var results = new List<Tensor>();
            foreach (var recomputeFn in _recomputeFunctions)
            {
                var recomputed = recomputeFn();
                results.AddRange(recomputed);
            }

            _recomputed = true;

            // Save recomputed activations for potential reuse
            foreach (var tensor in results)
            {
                _savedActivations.Add(tensor);
            }

            return results.ToArray();
        }
    }

    /// <summary>
    /// Gets saved activations by index.
    /// </summary>
    /// <param name="index">The index of the activation.</param>
    /// <returns>The saved activation.</returns>
    public Tensor GetSavedActivation(int index)
    {
        lock (_lock)
        {
            if (index < 0 || index >= _savedActivations.Count)
                throw new ArgumentOutOfRangeException(nameof(index), "Activation index out of range");

            return _savedActivations[index];
        }
    }

    /// <summary>
    /// Gets all saved activations.
    /// </summary>
    /// <returns>An array of all saved activations.</returns>
    public Tensor[] GetAllSavedActivations()
    {
        lock (_lock)
        {
            return _savedActivations.ToArray();
        }
    }

    /// <summary>
    /// Clears all saved activations to free memory.
    /// </summary>
    public void ClearSavedActivations()
    {
        lock (_lock)
        {
            _savedActivations.Clear();
        }
    }

    /// <summary>
    /// Gets the total size in bytes of saved activations.
    /// </summary>
    /// <returns>The memory size in bytes.</returns>
    public long GetSavedMemorySize()
    {
        lock (_lock)
        {
            return _savedActivations.Sum(t => (long)t.Size * sizeof(float));
        }
    }

    /// <summary>
    /// Gets the count of saved activations.
    /// </summary>
    public int SavedActivationCount
    {
        get
        {
            lock (_lock)
            {
                return _savedActivations.Count;
            }
        }
    }

    /// <summary>
    /// Gets whether this node has saved activations available.
    /// </summary>
    public bool HasSavedActivations
    {
        get
        {
            lock (_lock)
            {
                return _savedActivations.Count > 0;
            }
        }
    }

    /// <summary>
    /// Gets whether this node has recompute functions available.
    /// </summary>
    public bool HasRecomputeFunction
    {
        get
        {
            lock (_lock)
            {
                return _recomputeFunctions.Count > 0;
            }
        }
    }

    /// <summary>
    /// Sets the checkpoint scope for this node.
    /// </summary>
    /// <param name="scope">The checkpoint scope to associate with this node.</param>
    public void SetScope(CheckpointScope scope)
    {
        if (scope == null)
            throw new ArgumentNullException(nameof(scope));

        lock (_lock)
        {
            Scope = scope;
            IsCheckpoint = scope.IsEnabled;
            scope.RegisterNode(this);
        }
    }
}
