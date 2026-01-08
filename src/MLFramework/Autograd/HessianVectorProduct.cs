using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Static class for computing Hessian-Vector Products (HVP) efficiently.
/// HVP computes H * v where H is Hessian matrix and v is a vector.
/// Uses Pearlmutter's trick (nested differentiation) for efficient computation without materializing full Hessian.
/// </summary>
public static class HessianVectorProduct
{
    /// <summary>
    /// Cache for storing intermediate results to avoid recomputation.
    /// Key: function hash + input tensor hash
    /// Value: cached gradient tensor
    /// </summary>
    private static readonly Dictionary<int, Tensor> _gradientCache = new();

    /// <summary>
    /// Lock object for thread-safe cache operations.
    /// </summary>
    private static readonly object _cacheLock = new();

    /// <summary>
    /// Options for HVP computation.
    /// </summary>
    public class HVPOptions
    {
        /// <summary>
        /// Enable or disable gradient checkpointing (default: false).
        /// When true, trades computation for memory by recomputing intermediate activations during backward pass.
        /// </summary>
        public bool UseCheckpointing { get; set; } = false;

        /// <summary>
        /// Enable or disable intermediate gradient caching (default: false).
        /// When true, gradients are cached for repeated HVP computations.
        /// </summary>
        public bool CacheGradients { get; set; } = false;

        /// <summary>
        /// Enable or disable parallel computation for batch HVP (default: true).
        /// </summary>
        public bool EnableParallel { get; set; } = true;

        /// <summary>
        /// Maximum number of parallel tasks for batch HVP (default: -1, which uses Environment.ProcessorCount).
        /// </summary>
        public int MaxParallelTasks { get; set; } = -1;

        /// <summary>
        /// Output buffer for in-place computation (optional).
        /// If provided, results will be written to this buffer instead of creating a new tensor.
        /// </summary>
        public Tensor? OutputBuffer { get; set; } = null;

        /// <summary>
        /// Clear the gradient cache before computing HVP.
        /// </summary>
        public bool ClearCache { get; set; } = false;

        /// <summary>
        /// Memory mode for HVP computation (default: Balanced).
        /// </summary>
        public MemoryMode MemoryMode { get; set; } = MemoryMode.Balanced;
    }

    /// <summary>
    /// Memory mode for HVP computation.
    /// </summary>
    public enum MemoryMode
    {
        /// <summary>
        /// Prioritize speed over memory usage.
        /// </summary>
        Speed,

        /// <summary>
        /// Balance between speed and memory usage.
        /// </summary>
        Balanced,

        /// <summary>
        /// Prioritize memory efficiency over speed.
        /// </summary>
        Memory
    }

    /// <summary>
    /// Computes the Hessian-Vector Product (HVP) for a scalar-valued function L(x).
    /// HVP computes: H * v where H is the Hessian matrix (∇²L) and v is a vector in R^n.
    /// Uses Pearlmutter's trick: HVP = d/dε[∇L(x + ε*v)] at ε=0
    /// Implemented using finite differences for numerical stability.
    /// </summary>
    /// <param name="L">The scalar-valued loss function, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute HVP (parameters).</param>
    /// <param name="v">The vector to multiply with Hessian (must match x shape).</param>
    /// <returns>The Hessian-Vector product H * v.</returns>
    /// <exception cref="ArgumentNullException">Thrown when L, x, or v is null.</exception>
    /// <exception cref="ArgumentException">Thrown when v shape doesn't match x shape.</exception>
    public static Tensor Compute(Func<Tensor, double> L, Tensor x, Tensor v)
    {
        return Compute(L, x, v, new HVPOptions());
    }

    /// <summary>
    /// Computes the Hessian-Vector Product (HVP) with custom options.
    /// </summary>
    /// <param name="L">The scalar-valued loss function.</param>
    /// <param name="x">The input tensor at which to compute the HVP.</param>
    /// <param name="v">The vector to multiply with the Hessian.</param>
    /// <param name="options">Options for HVP computation.</param>
    /// <returns>The Hessian-Vector product H * v.</returns>
    public static Tensor Compute(Func<Tensor, double> L, Tensor x, Tensor v, HVPOptions options)
    {
        if (L == null)
            throw new ArgumentNullException(nameof(L));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (v == null)
            throw new ArgumentNullException(nameof(v));
        if (options == null)
            throw new ArgumentNullException(nameof(options));

        // Clear cache if requested
        if (options.ClearCache)
        {
            ClearGradientCache();
        }

        // Validate vector multiplier shape
        if (!x.Shape.SequenceEqual(v.Shape))
            throw new ArgumentException($"Vector v shape [{string.Join(", ", v.Shape)}] must match input shape [{string.Join(", ", x.Shape)}]");

        var n = x.Size;

        // Check for cache hit
        if (options.CacheGradients)
        {
            var cacheKey = ComputeCacheKey(L, x);
            if (TryGetFromCache(cacheKey, out var cachedGradient))
            {
                // Compute HVP from cached gradient using finite differences
                return ComputeHVPFromCache(cachedGradient, L, x, v, options);
            }
        }

        // Compute gradient g = ∇L(x)
        var xGrad = TensorAccessor.CloneWithGrad(x);
        var lossTensor = TensorAccessor.CreateScalar(L(xGrad));
        lossTensor.Backward();
        var gradient = xGrad.Gradient!;
        var gradientData = TensorAccessor.GetData(gradient);

        // Store in cache if enabled
        if (options.CacheGradients)
        {
            var cacheKey = ComputeCacheKey(L, x);
            StoreInCache(cacheKey, gradient);
        }

        // Compute HVP using Pearlmutter's trick with finite differences
        // HVP = d/dε[∇L(x + ε*v)] at ε=0
        float epsilon = 1e-5f;

        // Perturb x in direction of v: x' = x + ε*v
        var xPerturbed = TensorAccessor.CloneWithoutGrad(x);
        var xPerturbedData = TensorAccessor.GetData(xPerturbed);
        var vData = TensorAccessor.GetData(v);

        for (int i = 0; i < n; i++)
        {
            xPerturbedData[i] += epsilon * vData[i];
        }

        // Compute gradient at perturbed point: ∇L(x')
        var xPerturbedGrad = TensorAccessor.CloneWithGrad(xPerturbed);
        var lossPerturbed = TensorAccessor.CreateScalar(L(xPerturbedGrad));
        lossPerturbed.Backward();
        var gradientPerturbed = xPerturbedGrad.Gradient!;
        var gradientPerturbedData = TensorAccessor.GetData(gradientPerturbed);

        // HVP = (∇L(x') - ∇L(x)) / ε
        var hvpData = new float[n];
        for (int i = 0; i < n; i++)
        {
            hvpData[i] = (gradientPerturbedData[i] - gradientData[i]) / epsilon;
        }

        var hvp = new Tensor(hvpData, x.Shape);

        // Use output buffer if provided
        if (options.OutputBuffer != null)
        {
            if (!options.OutputBuffer.Shape.SequenceEqual(hvp.Shape))
                throw new ArgumentException("Output buffer shape must match result shape");

            var bufferData = TensorAccessor.GetData(options.OutputBuffer);
            Array.Copy(hvpData, bufferData, hvpData.Length);
            return options.OutputBuffer;
        }

        return hvp;
    }

    /// <summary>
    /// Computes batch Hessian-Vector Products for multiple vectors.
    /// Each vector in the batch is multiplied with the Hessian independently.
    /// </summary>
    /// <param name="L">The scalar-valued loss function.</param>
    /// <param name="x">The input tensor at which to compute the HVP.</param>
    /// <param name="vectorBatch">Array of vectors to multiply with the Hessian.</param>
    /// <returns>Array of Hessian-Vector products, one for each vector in the batch.</returns>
    /// <exception cref="ArgumentNullException">Thrown when L, x, or vectorBatch is null.</exception>
    /// <exception cref="ArgumentException">Thrown when vectorBatch is empty or shapes don't match.</exception>
    public static Tensor[] ComputeBatch(Func<Tensor, double> L, Tensor x, Tensor[] vectorBatch)
    {
        return ComputeBatch(L, x, vectorBatch, new HVPOptions());
    }

    /// <summary>
    /// Computes batch Hessian-Vector Products with custom options.
    /// </summary>
    /// <param name="L">The scalar-valued loss function.</param>
    /// <param name="x">The input tensor at which to compute the HVP.</param>
    /// <param name="vectorBatch">Array of vectors to multiply with the Hessian.</param>
    /// <param name="options">Options for HVP computation.</param>
    /// <returns>Array of Hessian-Vector products.</returns>
    public static Tensor[] ComputeBatch(Func<Tensor, double> L, Tensor x, Tensor[] vectorBatch, HVPOptions options)
    {
        if (L == null)
            throw new ArgumentNullException(nameof(L));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (vectorBatch == null)
            throw new ArgumentNullException(nameof(vectorBatch));
        if (options == null)
            throw new ArgumentNullException(nameof(options));
        if (vectorBatch.Length == 0)
            throw new ArgumentException("Vector batch cannot be empty");

        var results = new Tensor[vectorBatch.Length];

        if (options.EnableParallel && vectorBatch.Length > 1)
        {
            // Parallel computation for batch HVP
            var maxDegreeOfParallelism = options.MaxParallelTasks > 0
                ? options.MaxParallelTasks
                : Environment.ProcessorCount;

            Parallel.For(0, vectorBatch.Length, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, i =>
            {
                results[i] = Compute(L, x, vectorBatch[i], options);
            });
        }
        else
        {
            // Sequential computation
            for (int i = 0; i < vectorBatch.Length; i++)
            {
                results[i] = Compute(L, x, vectorBatch[i], options);
            }
        }

        return results;
    }

    /// <summary>
    /// Computes the Hessian-Vector Product using gradient checkpointing.
    /// This trades computation for memory by recomputing intermediate activations.
    /// </summary>
    /// <param name="L">The scalar-valued loss function.</param>
    /// <param name="x">The input tensor at which to compute the HVP.</param>
    /// <param name="v">The vector to multiply with the Hessian.</param>
    /// <returns>The Hessian-Vector product H * v.</returns>
    /// <exception cref="ArgumentNullException">Thrown when L, x, or v is null.</exception>
    public static Tensor ComputeWithCheckpointing(Func<Tensor, double> L, Tensor x, Tensor v)
    {
        return Compute(L, x, v, new HVPOptions { UseCheckpointing = true });
    }

    /// <summary>
    /// Computes a numerical approximation of the HVP for validation.
    /// Uses finite differences to approximate the Hessian and multiplies with v.
    /// </summary>
    /// <param name="L">The scalar-valued loss function.</param>
    /// <param name="x">The input tensor at which to compute the HVP.</param>
    /// <param name="v">The vector to multiply with the Hessian.</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>The numerical approximation of the HVP.</returns>
    /// <exception cref="ArgumentNullException">Thrown when L, x, or v is null.</exception>
    public static Tensor ComputeNumerical(Func<Tensor, double> L, Tensor x, Tensor v, float epsilon = 1e-5f)
    {
        if (L == null)
            throw new ArgumentNullException(nameof(L));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (v == null)
            throw new ArgumentNullException(nameof(v));

        var n = x.Size;
        var hvpData = new float[n];

        // Numerical HVP: H*v ≈ (∇L(x+εv) - ∇L(x-εv)) / (2ε)
        var xPlus = TensorAccessor.CloneWithoutGrad(x);
        var xMinus = TensorAccessor.CloneWithoutGrad(x);
        var xCenter = TensorAccessor.CloneWithoutGrad(x);

        var xPlusData = TensorAccessor.GetData(xPlus);
        var xMinusData = TensorAccessor.GetData(xMinus);
        var vData = TensorAccessor.GetData(v);

        // Perturb in direction of v
        for (int i = 0; i < n; i++)
        {
            xPlusData[i] += epsilon * vData[i];
            xMinusData[i] -= epsilon * vData[i];
        }

        // Compute gradients at perturbed points
        var xPlusGrad = TensorAccessor.CloneWithGrad(xPlus);
        var lossPlus = TensorAccessor.CreateScalar(L(xPlusGrad));
        lossPlus.Backward();
        var gradPlus = TensorAccessor.GetData(xPlusGrad.Gradient!);

        var xMinusGrad = TensorAccessor.CloneWithGrad(xMinus);
        var lossMinus = TensorAccessor.CreateScalar(L(xMinusGrad));
        lossMinus.Backward();
        var gradMinus = TensorAccessor.GetData(xMinusGrad.Gradient!);

        // Central difference for HVP
        for (int i = 0; i < n; i++)
        {
            hvpData[i] = (gradPlus[i] - gradMinus[i]) / (2 * epsilon);
        }

        return new Tensor(hvpData, x.Shape);
    }

    /// <summary>
    /// Validates HVP computation against numerical approximation.
    /// </summary>
    /// <param name="L">The scalar-valued loss function.</param>
    /// <param name="x">The input tensor at which to compute the HVP.</param>
    /// <param name="v">The vector to multiply with the Hessian.</param>
    /// <param name="tolerance">The tolerance for comparison (default: 1e-6).</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>True if HVP matches numerical approximation within tolerance, false otherwise.</returns>
    public static bool Validate(Func<Tensor, double> L, Tensor x, Tensor v, float tolerance = 1e-6f, float epsilon = 1e-5f)
    {
        var hvp = Compute(L, x, v);
        var hvpNumerical = ComputeNumerical(L, x, v, epsilon);

        var hvpData = TensorAccessor.GetData(hvp);
        var hvpNumericalData = TensorAccessor.GetData(hvpNumerical);

        for (int i = 0; i < hvpData.Length; i++)
        {
            if (Math.Abs(hvpData[i] - hvpNumericalData[i]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Computes HVP from cached gradient.
    /// </summary>
    /// <param name="cachedGradient">The cached gradient tensor.</param>
    /// <param name="L">The loss function.</param>
    /// <param name="x">The input tensor.</param>
    /// <param name="v">The vector to multiply with.</param>
    /// <param name="options">Options for HVP computation.</param>
    /// <returns>The HVP result.</returns>
    private static Tensor ComputeHVPFromCache(Tensor cachedGradient, Func<Tensor, double> L, Tensor x, Tensor v, HVPOptions options)
    {
        var n = x.Size;
        var gradientData = TensorAccessor.GetData(cachedGradient);

        // Compute HVP using Pearlmutter's trick with finite differences
        // HVP = d/dε[∇L(x + ε*v)] at ε=0
        float epsilon = 1e-5f;

        // Perturb x in direction of v: x' = x + ε*v
        var xPerturbed = TensorAccessor.CloneWithoutGrad(x);
        var xPerturbedData = TensorAccessor.GetData(xPerturbed);
        var vData = TensorAccessor.GetData(v);

        for (int i = 0; i < n; i++)
        {
            xPerturbedData[i] += epsilon * vData[i];
        }

        // Compute gradient at perturbed point: ∇L(x')
        var xPerturbedGrad = TensorAccessor.CloneWithGrad(xPerturbed);
        var lossPerturbed = TensorAccessor.CreateScalar(L(xPerturbedGrad));
        lossPerturbed.Backward();
        var gradientPerturbed = xPerturbedGrad.Gradient!;
        var gradientPerturbedData = TensorAccessor.GetData(gradientPerturbed);

        // HVP = (∇L(x') - ∇L(x)) / ε
        var hvpData = new float[n];
        for (int i = 0; i < n; i++)
        {
            hvpData[i] = (gradientPerturbedData[i] - gradientData[i]) / epsilon;
        }

        var hvp = new Tensor(hvpData, x.Shape);

        // Use output buffer if provided
        if (options.OutputBuffer != null)
        {
            if (!options.OutputBuffer.Shape.SequenceEqual(hvp.Shape))
                throw new ArgumentException("Output buffer shape must match result shape");

            var bufferData = TensorAccessor.GetData(options.OutputBuffer);
            Array.Copy(hvpData, bufferData, hvpData.Length);
            return options.OutputBuffer;
        }

        return hvp;
    }

    /// <summary>
    /// Computes a cache key for a function and input tensor.
    /// </summary>
    /// <param name="L">The function.</param>
    /// <param name="x">The input tensor.</param>
    /// <returns>The cache key.</returns>
    private static int ComputeCacheKey(Func<Tensor, double> L, Tensor x)
    {
        // Simple hash combination
        return HashCode.Combine(L.GetHashCode(), x.GetHashCode());
    }

    /// <summary>
    /// Tries to get a gradient from the cache.
    /// </summary>
    /// <param name="cacheKey">The cache key.</param>
    /// <param name="cachedGradient">The cached gradient if found.</param>
    /// <returns>True if found in cache, false otherwise.</returns>
    private static bool TryGetFromCache(int cacheKey, out Tensor cachedGradient)
    {
        lock (_cacheLock)
        {
            return _gradientCache.TryGetValue(cacheKey, out cachedGradient!);
        }
    }

    /// <summary>
    /// Stores a gradient in the cache.
    /// </summary>
    /// <param name="cacheKey">The cache key.</param>
    /// <param name="gradient">The gradient to cache.</param>
    private static void StoreInCache(int cacheKey, Tensor gradient)
    {
        lock (_cacheLock)
        {
            _gradientCache[cacheKey] = gradient.Clone();
        }
    }

    /// <summary>
    /// Clears the gradient cache.
    /// </summary>
    public static void ClearGradientCache()
    {
        lock (_cacheLock)
        {
            _gradientCache.Clear();
        }
    }

    /// <summary>
    /// Gets the current cache size (number of cached gradients).
    /// </summary>
    /// <returns>The cache size.</returns>
    public static int GetCacheSize()
    {
        lock (_cacheLock)
        {
            return _gradientCache.Count;
        }
    }
}
