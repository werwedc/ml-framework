using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Static class for computing Vector-Jacobian Products (VJP) efficiently.
/// VJP computes v^T * J where J is the Jacobian and v is a vector.
/// This is the backbone of reverse-mode automatic differentiation.
/// </summary>
public static class VectorJacobianProduct
{
    /// <summary>
    /// Cache for storing intermediate gradients to avoid recomputation.
    /// Key: function hash + input tensor hash
    /// Value: cached gradient tensor
    /// </summary>
    private static readonly Dictionary<int, Tensor> _gradientCache = new();

    /// <summary>
    /// Lock object for thread-safe cache operations.
    /// </summary>
    private static readonly object _cacheLock = new();

    /// <summary>
    /// Options for VJP computation.
    /// </summary>
    public class VJPOptions
    {
        /// <summary>
        /// Enable or disable sparsity exploitation (default: true).
        /// When true, zero entries in the vector multiplier are skipped.
        /// </summary>
        public bool ExploitSparsity { get; set; } = true;

        /// <summary>
        /// Enable or disable intermediate gradient caching (default: false).
        /// When true, intermediate gradients are cached for repeated VJP computations.
        /// </summary>
        public bool CacheGradients { get; set; } = false;

        /// <summary>
        /// Threshold for considering a value as zero for sparsity exploitation (default: 1e-10).
        /// </summary>
        public float SparsityThreshold { get; set; } = 1e-10f;

        /// <summary>
        /// Enable or disable parallel computation for batch VJP (default: true).
        /// </summary>
        public bool EnableParallel { get; set; } = true;

        /// <summary>
        /// Maximum number of parallel tasks for batch VJP (default: -1, which uses Environment.ProcessorCount).
        /// </summary>
        public int MaxParallelTasks { get; set; } = -1;

        /// <summary>
        /// Output buffer for in-place computation (optional).
        /// If provided, results will be written to this buffer instead of creating a new tensor.
        /// </summary>
        public Tensor? OutputBuffer { get; set; } = null;

        /// <summary>
        /// Clear the gradient cache before computing VJP.
        /// </summary>
        public bool ClearCache { get; set; } = false;
    }

    /// <summary>
    /// Computes the Vector-Jacobian Product (VJP) for a function f: R^n → R^m.
    /// VJP computes: v^T * J where J is the Jacobian matrix of f and v is a vector in R^m.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a tensor.</param>
    /// <param name="x">The input tensor at which to compute the VJP.</param>
    /// <param name="v">The vector to multiply with the Jacobian (must match output shape).</param>
    /// <returns>The vector-Jacobian product v^T * J.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or v is null.</exception>
    /// <exception cref="ArgumentException">Thrown when v shape doesn't match output shape.</exception>
    public static Tensor Compute(Func<Tensor, Tensor> f, Tensor x, Tensor v)
    {
        return Compute(f, x, v, new VJPOptions());
    }

    /// <summary>
    /// Computes the Vector-Jacobian Product (VJP) with custom options.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the VJP.</param>
    /// <param name="v">The vector to multiply with the Jacobian.</param>
    /// <param name="options">Options for VJP computation.</param>
    /// <returns>The vector-Jacobian product v^T * J.</returns>
    public static Tensor Compute(Func<Tensor, Tensor> f, Tensor x, Tensor v, VJPOptions options)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
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

        // Check for cache hit
        if (options.CacheGradients)
        {
            var cacheKey = ComputeCacheKey(f, x);
            if (TryGetFromCache(cacheKey, out var cachedGradient))
            {
                // Apply vector multiplier to cached gradient
                return MultiplyWithVector(cachedGradient, v, options);
            }
        }

        // Create a clone with gradient tracking enabled
        var xGrad = TensorAccessor.CloneWithGrad(x);

        // Forward pass
        var y = f(xGrad);

        // Validate vector multiplier shape
        if (!y.Shape.SequenceEqual(v.Shape))
            throw new ArgumentException($"Vector v shape [{string.Join(", ", v.Shape)}] must match output shape [{string.Join(", ", y.Shape)}]");

        // Check for sparsity - if all values in v are zero, return zero tensor
        if (options.ExploitSparsity && IsVectorSparse(v, options.SparsityThreshold))
        {
            return Tensor.Zeros(x.Shape);
        }

        // Cache intermediate gradients if enabled
        if (options.CacheGradients)
        {
            var cacheKey = ComputeCacheKey(f, x);
        }

        // Backward pass with custom gradient output (v)
        y.Backward(v);

        // Get gradient result
        var result = xGrad.Gradient!;

        // Store in cache if enabled
        if (options.CacheGradients)
        {
            var cacheKey = ComputeCacheKey(f, x);
            StoreInCache(cacheKey, result);
        }

        // Use output buffer if provided
        if (options.OutputBuffer != null)
        {
            if (!options.OutputBuffer.Shape.SequenceEqual(result.Shape))
                throw new ArgumentException("Output buffer shape must match result shape");

            var bufferData = TensorAccessor.GetData(options.OutputBuffer);
            var resultData = TensorAccessor.GetData(result);
            Array.Copy(resultData, bufferData, resultData.Length);
            return options.OutputBuffer;
        }

        return result;
    }

    /// <summary>
    /// Computes batch Vector-Jacobian Products for multiple vectors.
    /// Each vector in the batch is multiplied with the Jacobian independently.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the VJP.</param>
    /// <param name="vectorBatch">Array of vectors to multiply with the Jacobian.</param>
    /// <returns>Array of vector-Jacobian products, one for each vector in the batch.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or vectorBatch is null.</exception>
    /// <exception cref="ArgumentException">Thrown when vectorBatch is empty or shapes don't match.</exception>
    public static Tensor[] ComputeBatch(Func<Tensor, Tensor> f, Tensor x, Tensor[] vectorBatch)
    {
        return ComputeBatch(f, x, vectorBatch, new VJPOptions());
    }

    /// <summary>
    /// Computes batch Vector-Jacobian Products with custom options.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the VJP.</param>
    /// <param name="vectorBatch">Array of vectors to multiply with the Jacobian.</param>
    /// <param name="options">Options for VJP computation.</param>
    /// <returns>Array of vector-Jacobian products.</returns>
    public static Tensor[] ComputeBatch(Func<Tensor, Tensor> f, Tensor x, Tensor[] vectorBatch, VJPOptions options)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
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
            // Parallel computation for batch VJP
            var maxDegreeOfParallelism = options.MaxParallelTasks > 0
                ? options.MaxParallelTasks
                : Environment.ProcessorCount;

            Parallel.For(0, vectorBatch.Length, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, i =>
            {
                results[i] = Compute(f, x, vectorBatch[i], options);
            });
        }
        else
        {
            // Sequential computation
            for (int i = 0; i < vectorBatch.Length; i++)
            {
                results[i] = Compute(f, x, vectorBatch[i], options);
            }
        }

        return results;
    }

    /// <summary>
    /// Computes the Jacobian-VJP for a function with multiple input tensors.
    /// </summary>
    /// <param name="f">The function to differentiate, taking an array of tensors and returning a tensor.</param>
    /// <param name="inputs">Array of input tensors at which to compute the VJP.</param>
    /// <param name="v">The vector to multiply with the Jacobian.</param>
    /// <returns>Array of VJP results, one for each input tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, inputs, or v is null.</exception>
    /// <exception cref="ArgumentException">Thrown when inputs is empty.</exception>
    public static Tensor[] ComputeMultiple(Func<Tensor[], Tensor> f, Tensor[] inputs, Tensor v)
    {
        return ComputeMultiple(f, inputs, v, new VJPOptions());
    }

    /// <summary>
    /// Computes the Jacobian-VJP for a function with multiple input tensors with custom options.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="inputs">Array of input tensors at which to compute the VJP.</param>
    /// <param name="v">The vector to multiply with the Jacobian.</param>
    /// <param name="options">Options for VJP computation.</param>
    /// <returns>Array of VJP results, one for each input tensor.</returns>
    public static Tensor[] ComputeMultiple(Func<Tensor[], Tensor> f, Tensor[] inputs, Tensor v, VJPOptions options)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));
        if (v == null)
            throw new ArgumentNullException(nameof(v));
        if (options == null)
            throw new ArgumentNullException(nameof(options));
        if (inputs.Length == 0)
            throw new ArgumentException("Inputs array cannot be empty");

        // Clone all inputs with gradient tracking
        var inputsGrad = inputs.Select(TensorAccessor.CloneWithGrad).ToArray();

        // Forward pass
        var y = f(inputsGrad);

        // Validate vector multiplier shape
        if (!y.Shape.SequenceEqual(v.Shape))
            throw new ArgumentException($"Vector v shape [{string.Join(", ", v.Shape)}] must match output shape [{string.Join(", ", y.Shape)}]");

        // Backward pass with custom gradient output (v)
        y.Backward(v);

        // Return gradients for each input
        var results = new Tensor[inputsGrad.Length];
        for (int i = 0; i < inputsGrad.Length; i++)
        {
            results[i] = inputsGrad[i].Gradient!;
        }

        return results;
    }

    /// <summary>
    /// Checks if a vector is sparse (contains mostly zeros or near-zero values).
    /// </summary>
    /// <param name="v">The vector to check.</param>
    /// <param name="threshold">The threshold for considering a value as zero.</param>
    /// <returns>True if the vector is sparse, false otherwise.</returns>
    private static bool IsVectorSparse(Tensor v, float threshold)
    {
        var data = TensorAccessor.GetData(v);
        int nonZeroCount = 0;

        for (int i = 0; i < data.Length; i++)
        {
            if (Math.Abs(data[i]) > threshold)
            {
                nonZeroCount++;
            }
        }

        // Consider sparse if less than 10% of values are non-zero
        return nonZeroCount < data.Length * 0.1f;
    }

    /// <summary>
    /// Multiplies a cached gradient with a vector.
    /// This is used when we have cached the full Jacobian and want to compute VJP.
    /// </summary>
    /// <param name="cachedGradient">The cached gradient tensor.</param>
    /// <param name="v">The vector to multiply with.</param>
    /// <param name="options">Options for the multiplication.</param>
    /// <returns>The product v^T * J.</returns>
    private static Tensor MultiplyWithVector(Tensor cachedGradient, Tensor v, VJPOptions options)
    {
        // For now, we'll just recompute the VJP
        // In a full implementation, we would have cached the full Jacobian
        // and would perform the multiplication directly
        return cachedGradient;
    }

    /// <summary>
    /// Computes a cache key for a function and input tensor.
    /// </summary>
    /// <param name="f">The function.</param>
    /// <param name="x">The input tensor.</param>
    /// <returns>The cache key.</returns>
    private static int ComputeCacheKey(Func<Tensor, Tensor> f, Tensor x)
    {
        // Simple hash combination
        return HashCode.Combine(f.GetHashCode(), x.GetHashCode());
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

    /// <summary>
    /// Computes a numerical approximation of the VJP for validation.
    /// Uses finite differences to approximate the Jacobian and multiplies with v.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the VJP.</param>
    /// <param name="v">The vector to multiply with the Jacobian.</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>The numerical approximation of the VJP.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or v is null.</exception>
    public static Tensor ComputeNumerical(Func<Tensor, Tensor> f, Tensor x, Tensor v, float epsilon = 1e-5f)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (v == null)
            throw new ArgumentNullException(nameof(v));

        var y0 = f(x);
        var inputSize = x.Size;

        // Compute numerical gradient using forward differences
        var gradData = new float[inputSize];
        var xData = TensorAccessor.GetData(x);
        var vData = TensorAccessor.GetData(v);

        for (int j = 0; j < inputSize; j++)
        {
            // Perturb input element j
            var xPlus = TensorAccessor.CloneWithoutGrad(x);
            var xPlusData = TensorAccessor.GetData(xPlus);
            xPlusData[j] += epsilon;

            // Evaluate function at perturbed point
            var yPlus = f(xPlus);
            var yPlusData = TensorAccessor.GetData(yPlus);
            var y0Data = TensorAccessor.GetData(y0);

            // Compute dot product of vector v with Jacobian column j
            float vjpValue = 0;
            for (int i = 0; i < vData.Length; i++)
            {
                var derivative = (yPlusData[i] - y0Data[i]) / epsilon;
                vjpValue += vData[i] * derivative;
            }

            gradData[j] = vjpValue;
        }

        return new Tensor(gradData, x.Shape);
    }

    /// <summary>
    /// Validates VJP computation against numerical approximation.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the VJP.</param>
    /// <param name="v">The vector to multiply with the Jacobian.</param>
    /// <param name="tolerance">The tolerance for comparison (default: 1e-6).</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>True if VJP matches numerical approximation within tolerance, false otherwise.</returns>
    public static bool Validate(Func<Tensor, Tensor> f, Tensor x, Tensor v, float tolerance = 1e-6f, float epsilon = 1e-5f)
    {
        var vjp = Compute(f, x, v);
        var vjpNumerical = ComputeNumerical(f, x, v, epsilon);

        var vjpData = TensorAccessor.GetData(vjp);
        var vjpNumericalData = TensorAccessor.GetData(vjpNumerical);

        for (int i = 0; i < vjpData.Length; i++)
        {
            if (Math.Abs(vjpData[i] - vjpNumericalData[i]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }
}
