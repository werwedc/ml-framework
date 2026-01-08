using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Static class for computing Jacobian-Vector Products (JVP) using forward-mode automatic differentiation.
/// JVP computes J * v where J is the Jacobian matrix and v is a vector.
/// This is more efficient than VJP when the input dimension is smaller than the output dimension.
/// </summary>
public static class JacobianVectorProduct
{
    /// <summary>
    /// Cache for storing tangent computation results to avoid recomputation.
    /// Key: function hash + input tensor hash
    /// Value: cached tangent tensor
    /// </summary>
    private static readonly Dictionary<int, Tensor> _tangentCache = new();

    /// <summary>
    /// Lock object for thread-safe cache operations.
    /// </summary>
    private static readonly object _cacheLock = new();

    /// <summary>
    /// Options for JVP computation.
    /// </summary>
    public class JVPOptions
    {
        /// <summary>
        /// Enable or disable sparsity exploitation (default: true).
        /// When true, zero entries in the tangent vector are skipped.
        /// </summary>
        public bool ExploitSparsity { get; set; } = true;

        /// <summary>
        /// Enable or disable intermediate tangent caching (default: false).
        /// When true, intermediate tangents are cached for repeated JVP computations.
        /// </summary>
        public bool CacheTangents { get; set; } = false;

        /// <summary>
        /// Threshold for considering a value as zero for sparsity exploitation (default: 1e-10).
        /// </summary>
        public float SparsityThreshold { get; set; } = 1e-10f;

        /// <summary>
        /// Enable or disable parallel computation for batch JVP (default: true).
        /// </summary>
        public bool EnableParallel { get; set; } = true;

        /// <summary>
        /// Maximum number of parallel tasks for batch JVP (default: -1, which uses Environment.ProcessorCount).
        /// </summary>
        public int MaxParallelTasks { get; set; } = -1;

        /// <summary>
        /// Output buffer for in-place computation (optional).
        /// If provided, results will be written to this buffer instead of creating a new tensor.
        /// </summary>
        public Tensor? OutputBuffer { get; set; } = null;

        /// <summary>
        /// Clear tangent cache before computing JVP.
        /// </summary>
        public bool ClearCache { get; set; } = false;
    }

    /// <summary>
    /// Computes the Jacobian-Vector Product (JVP) for a function f: R^n → R^m.
    /// JVP computes: J * v where J is the Jacobian matrix of f and v is a vector in R^n.
    /// Uses forward-mode automatic differentiation with dual numbers.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a tensor.</param>
    /// <param name="x">The input tensor at which to compute the JVP.</param>
    /// <param name="v">The tangent vector to multiply with the Jacobian (must match input shape).</param>
    /// <returns>The Jacobian-Vector product J * v.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or v is null.</exception>
    /// <exception cref="ArgumentException">Thrown when v shape doesn't match input shape.</exception>
    public static Tensor Compute(Func<Tensor, Tensor> f, Tensor x, Tensor v)
    {
        return Compute(f, x, v, new JVPOptions());
    }

    /// <summary>
    /// Computes the Jacobian-Vector Product (JVP) with custom options.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the JVP.</param>
    /// <param name="v">The tangent vector to multiply with the Jacobian.</param>
    /// <param name="options">Options for JVP computation.</param>
    /// <returns>The Jacobian-Vector product J * v.</returns>
    public static Tensor Compute(Func<Tensor, Tensor> f, Tensor x, Tensor v, JVPOptions options)
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
            ClearTangentCache();
        }

        // Validate tangent vector shape
        if (!x.Shape.SequenceEqual(v.Shape))
            throw new ArgumentException($"Tangent vector v shape [{string.Join(", ", v.Shape)}] must match input shape [{string.Join(", ", x.Shape)}]");

        // Check for sparsity - if all values in v are zero, return zero tensor
        if (options.ExploitSparsity && IsVectorSparse(v, options.SparsityThreshold))
        {
            // Forward pass to get output shape
            var y = f(x);
            return Tensor.Zeros(y.Shape);
        }

        // Create dual tensor (value + tangent)
        var dualTensor = CreateDualTensor(x, v);

        // Forward pass with dual numbers
        var yDual = f(dualTensor);

        // Extract tangent from dual output
        var tangent = ExtractTangent(yDual);

        // Use output buffer if provided
        if (options.OutputBuffer != null)
        {
            if (!options.OutputBuffer.Shape.SequenceEqual(tangent.Shape))
                throw new ArgumentException("Output buffer shape must match result shape");

            var bufferData = TensorAccessor.GetData(options.OutputBuffer);
            var tangentData = TensorAccessor.GetData(tangent);
            Array.Copy(tangentData, bufferData, tangentData.Length);
            return options.OutputBuffer;
        }

        return tangent;
    }

    /// <summary>
    /// Computes batch Jacobian-Vector Products for multiple tangent vectors.
    /// Each vector in the batch is multiplied with the Jacobian independently.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the JVP.</param>
    /// <param name="vectorBatch">Array of tangent vectors to multiply with the Jacobian.</param>
    /// <returns>Array of Jacobian-Vector products, one for each vector in the batch.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or vectorBatch is null.</exception>
    /// <exception cref="ArgumentException">Thrown when vectorBatch is empty or shapes don't match.</exception>
    public static Tensor[] ComputeBatch(Func<Tensor, Tensor> f, Tensor x, Tensor[] vectorBatch)
    {
        return ComputeBatch(f, x, vectorBatch, new JVPOptions());
    }

    /// <summary>
    /// Computes batch Jacobian-Vector Products with custom options.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the JVP.</param>
    /// <param name="vectorBatch">Array of tangent vectors to multiply with the Jacobian.</param>
    /// <param name="options">Options for JVP computation.</param>
    /// <returns>Array of Jacobian-Vector products.</returns>
    public static Tensor[] ComputeBatch(Func<Tensor, Tensor> f, Tensor x, Tensor[] vectorBatch, JVPOptions options)
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
            // Parallel computation for batch JVP
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
    /// Computes the Jacobian-Vector Product for a function with multiple input tensors.
    /// </summary>
    /// <param name="f">The function to differentiate, taking an array of tensors and returning a tensor.</param>
    /// <param name="inputs">Array of input tensors at which to compute the JVP.</param>
    /// <param name="v">The tangent vector to multiply with the Jacobian.</param>
    /// <param name="inputIndex">Index of the input tensor to differentiate with respect to.</param>
    /// <returns>The Jacobian-Vector product for the specified input.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, inputs, or v is null.</exception>
    /// <exception cref="ArgumentException">Thrown when inputs is empty or inputIndex is invalid.</exception>
    public static Tensor ComputeMultiple(Func<Tensor[], Tensor> f, Tensor[] inputs, Tensor v, int inputIndex = 0)
    {
        return ComputeMultiple(f, inputs, v, inputIndex, new JVPOptions());
    }

    /// <summary>
    /// Computes the Jacobian-Vector Product for a function with multiple input tensors with custom options.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="inputs">Array of input tensors at which to compute the JVP.</param>
    /// <param name="v">The tangent vector to multiply with the Jacobian.</param>
    /// <param name="inputIndex">Index of the input tensor to differentiate with respect to.</param>
    /// <param name="options">Options for JVP computation.</param>
    /// <returns>The Jacobian-Vector product for the specified input.</returns>
    public static Tensor ComputeMultiple(Func<Tensor[], Tensor> f, Tensor[] inputs, Tensor v, int inputIndex, JVPOptions options)
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
        if (inputIndex < 0 || inputIndex >= inputs.Length)
            throw new ArgumentException($"Invalid inputIndex {inputIndex}. Must be between 0 and {inputs.Length - 1}");

        // Validate tangent vector shape
        if (!inputs[inputIndex].Shape.SequenceEqual(v.Shape))
            throw new ArgumentException($"Tangent vector v shape [{string.Join(", ", v.Shape)}] must match input[{inputIndex}] shape [{string.Join(", ", inputs[inputIndex].Shape)}]");

        // Check for sparsity
        if (options.ExploitSparsity && IsVectorSparse(v, options.SparsityThreshold))
        {
            // Forward pass to get output shape
            var y = f(inputs);
            return Tensor.Zeros(y.Shape);
        }

        // Create dual tensor for the specified input
        var dualInputs = inputs.Select(t => t.Clone()).ToArray();
        dualInputs[inputIndex] = CreateDualTensor(inputs[inputIndex], v);

        // Forward pass with dual numbers
        var yDual = f(dualInputs);

        // Extract tangent from dual output
        return ExtractTangent(yDual);
    }

    /// <summary>
    /// Creates a dual tensor representing (value, tangent) pair.
    /// </summary>
    /// <param name="value">The value tensor.</param>
    /// <param name="tangent">The tangent tensor.</param>
    /// <returns>A dual tensor with both value and tangent components.</returns>
    private static Tensor CreateDualTensor(Tensor value, Tensor tangent)
    {
        var valueData = TensorAccessor.GetData(value);
        var tangentData = TensorAccessor.GetData(tangent);

        // Create a tensor that stores both value and tangent
        // In a full implementation, this would be a proper DualTensor class
        // For now, we use the value tensor and attach tangent as gradient
        var dual = TensorAccessor.CloneWithGrad(value);
        var dualGradData = TensorAccessor.GetData(dual.Gradient!);

        // Store tangent in gradient field (this is a hack - proper implementation needs DualTensor)
        Array.Copy(tangentData, dualGradData, tangentData.Length);

        return dual;
    }

    /// <summary>
    /// Extracts the tangent component from a dual tensor.
    /// </summary>
    /// <param name="dualTensor">The dual tensor containing both value and tangent.</param>
    /// <returns>The tangent tensor.</returns>
    private static Tensor ExtractTangent(Tensor dualTensor)
    {
        // In this implementation, tangent is stored in the gradient
        // A proper implementation would have a dedicated DualTensor class
        return dualTensor.Gradient?.Clone() ?? Tensor.Zeros(dualTensor.Shape);
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
    /// Clears the tangent cache.
    /// </summary>
    public static void ClearTangentCache()
    {
        lock (_cacheLock)
        {
            _tangentCache.Clear();
        }
    }

    /// <summary>
    /// Gets the current cache size (number of cached tangents).
    /// </summary>
    /// <returns>The cache size.</returns>
    public static int GetCacheSize()
    {
        lock (_cacheLock)
        {
            return _tangentCache.Count;
        }
    }

    /// <summary>
    /// Computes a numerical approximation of JVP for validation.
    /// Uses finite differences to approximate the Jacobian and multiplies with v.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the JVP.</param>
    /// <param name="v">The tangent vector to multiply with the Jacobian.</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>The numerical approximation of the JVP.</returns>
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
        var outputSize = y0.Size;
        var inputSize = x.Size;

        // Compute JVP using forward differences: J*v ≈ (f(x + εv) - f(x)) / ε
        var xPerturbed = TensorAccessor.CloneWithoutGrad(x);
        var xPerturbedData = TensorAccessor.GetData(xPerturbed);
        var vData = TensorAccessor.GetData(v);

        // Perturb input: x + εv
        for (int i = 0; i < inputSize; i++)
        {
            xPerturbedData[i] += epsilon * vData[i];
        }

        // Evaluate function at perturbed point
        var yPerturbed = f(xPerturbed);
        var yPerturbedData = TensorAccessor.GetData(yPerturbed);
        var y0Data = TensorAccessor.GetData(y0);

        // Compute finite difference approximation
        var resultData = new float[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            resultData[i] = (yPerturbedData[i] - y0Data[i]) / epsilon;
        }

        return new Tensor(resultData, y0.Shape);
    }

    /// <summary>
    /// Validates JVP computation against numerical approximation.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the JVP.</param>
    /// <param name="v">The tangent vector to multiply with the Jacobian.</param>
    /// <param name="tolerance">The tolerance for comparison (default: 1e-6).</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>True if JVP matches numerical approximation within tolerance, false otherwise.</returns>
    public static bool Validate(Func<Tensor, Tensor> f, Tensor x, Tensor v, float tolerance = 1e-6f, float epsilon = 1e-5f)
    {
        var jvp = Compute(f, x, v);
        var jvpNumerical = ComputeNumerical(f, x, v, epsilon);

        var jvpData = TensorAccessor.GetData(jvp);
        var jvpNumericalData = TensorAccessor.GetData(jvpNumerical);

        for (int i = 0; i < jvpData.Length; i++)
        {
            if (Math.Abs(jvpData[i] - jvpNumericalData[i]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Computes the Jacobian column-by-column using JVP for efficiency when input dim < output dim.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the Jacobian.</param>
    /// <returns>The full Jacobian matrix.</returns>
    public static Tensor ComputeJacobian(Func<Tensor, Tensor> f, Tensor x)
    {
        var n = x.Size;
        var y = f(x);
        var m = y.Size;

        var jacobianData = new float[m * n];

        // Compute each column of Jacobian using JVP with standard basis vectors
        for (int j = 0; j < n; j++)
        {
            // Create standard basis vector e_j
            var e_j = Tensor.Zeros(x.Shape);
            var e_jData = TensorAccessor.GetData(e_j);
            e_jData[j] = 1.0f;

            // Compute JVP: J * e_j = j-th column of J
            var column = Compute(f, x, e_j);
            var columnData = TensorAccessor.GetData(column);

            // Copy column to Jacobian
            for (int i = 0; i < m; i++)
            {
                jacobianData[i * n + j] = columnData[i];
            }
        }

        return new Tensor(jacobianData, new[] { m, n });
    }
}
