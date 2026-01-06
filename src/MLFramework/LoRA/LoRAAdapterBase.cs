using MLFramework.Modules;
using RitterFramework.Core.Tensor;

namespace MLFramework.LoRA;

/// <summary>
/// Base class for LoRA (Low-Rank Adaptation) adapters.
/// </summary>
public abstract class LoRAAdapterBase : IModule
{
    protected readonly IModule _baseModule;
    protected readonly int _rank;
    protected readonly float _alpha;

    /// <summary>
    /// Initializes a new instance of the LoRAAdapterBase class.
    /// </summary>
    /// <param name="baseModule">The base module to adapt.</param>
    /// <param name="rank">The rank of the adaptation.</param>
    /// <param name="alpha">The alpha scaling factor.</param>
    protected LoRAAdapterBase(IModule baseModule, int rank, float alpha)
    {
        _baseModule = baseModule ?? throw new ArgumentNullException(nameof(baseModule));
        _rank = rank;
        _alpha = alpha;
    }

    /// <summary>
    /// Gets the module type.
    /// </summary>
    public string ModuleType => "LoRAAdapter";

    /// <summary>
    /// Gets the scaling factor (alpha/rank).
    /// </summary>
    public float ScalingFactor => _alpha / _rank;

    /// <summary>
    /// Gets the rank of the adaptation.
    /// </summary>
    public int Rank => _rank;

    /// <summary>
    /// Gets the alpha scaling factor.
    /// </summary>
    public int Alpha => (int)_alpha;

    /// <summary>
    /// Gets trainable parameters.
    /// </summary>
    public abstract IEnumerable<Tensor> TrainableParameters { get; }

    /// <summary>
    /// Gets frozen parameters.
    /// </summary>
    public abstract IEnumerable<Tensor> FrozenParameters { get; }

    /// <summary>
    /// Freezes the base layer.
    /// </summary>
    public abstract void FreezeBaseLayer();

    /// <summary>
    /// Unfreezes the base layer.
    /// </summary>
    public abstract void UnfreezeBaseLayer();

    /// <summary>
    /// Merges the adapter weights into the base layer.
    /// </summary>
    public abstract void MergeAdapter();

    /// <summary>
    /// Resets the base layer to original weights.
    /// </summary>
    public abstract void ResetBaseLayer();

    /// <summary>
    /// Gets adapter weights.
    /// </summary>
    /// <returns>Tuple of (MatrixA, MatrixB).</returns>
    public abstract (Tensor? MatrixA, Tensor? MatrixB) GetAdapterWeights();

    /// <summary>
    /// Sets adapter weights.
    /// </summary>
    /// <param name="matrixA">Matrix A weights.</param>
    /// <param name="matrixB">Matrix B weights.</param>
    public abstract void SetAdapterWeights(Tensor? matrixA, Tensor? matrixB);
}
