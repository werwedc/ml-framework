using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Specifies the computation mode for Jacobian calculation.
/// </summary>
public enum JacobianMode
{
    /// <summary>
    /// Automatically select the most efficient mode based on input/output dimensions.
    /// Uses VJP (reverse mode) when output dimension < input dimension.
    /// Uses JVP (forward mode) when input dimension < output dimension.
    /// </summary>
    Auto,

    /// <summary>
    /// Force use of forward-mode automatic differentiation (JVP).
    /// More efficient when input dimension is smaller than output dimension.
    /// </summary>
    Forward,

    /// <summary>
    /// Force use of reverse-mode automatic differentiation (VJP).
    /// More efficient when output dimension is smaller than input dimension.
    /// </summary>
    Reverse
}

/// <summary>
/// Specifies the structure type detected in the Jacobian matrix.
/// </summary>
public enum JacobianStructure
{
    /// <summary>
    /// No special structure detected (general dense matrix).
    /// </summary>
    General,

    /// <summary>
    /// Jacobian is a diagonal matrix (element-wise operations).
    /// </summary>
    Diagonal,

    /// <summary>
    /// Jacobian has a block-diagonal structure.
    /// </summary>
    BlockDiagonal,

    /// <summary>
    /// Jacobian is triangular (upper or lower).
    /// </summary>
    Triangular
}

/// <summary>
/// Options for computing the full Jacobian matrix.
/// </summary>
public class JacobianOptions
{
    /// <summary>
    /// The computation mode for Jacobian calculation (default: Auto).
    /// </summary>
    public JacobianMode Mode { get; set; } = JacobianMode.Auto;

    /// <summary>
    /// Whether to compute a sparse Jacobian (default: false).
    /// When true, returns a sparse tensor representation to save memory.
    /// </summary>
    public bool Sparse { get; set; } = false;

    /// <summary>
    /// Threshold for considering a value as zero in sparse representation (default: 1e-10).
    /// Values with absolute value below this threshold are treated as zero.
    /// </summary>
    public float SparsityThreshold { get; set; } = 1e-10f;

    /// <summary>
    /// Whether to enable automatic structure detection (default: true).
    /// When true, the algorithm detects special structures like diagonal, block-diagonal, etc.
    /// </summary>
    public bool DetectStructure { get; set; } = true;

    /// <summary>
    /// Whether to use parallel computation for independent columns/rows (default: true).
    /// </summary>
    public bool EnableParallel { get; set; } = true;

    /// <summary>
    /// Maximum number of parallel tasks for computation (default: -1, uses Environment.ProcessorCount).
    /// </summary>
    public int MaxParallelTasks { get; set; } = -1;

    /// <summary>
    /// Output indices for partial Jacobian computation (optional).
    /// If specified, only computes Jacobian for these output indices.
    /// </summary>
    public int[]? OutputIndices { get; set; } = null;

    /// <summary>
    /// Progress callback for long-running Jacobian computations (optional).
    /// Called with (completed, total) progress information.
    /// </summary>
    public Action<int, int>? ProgressCallback { get; set; } = null;

    /// <summary>
    /// Whether to warn about potentially expensive computations (default: true).
    /// </summary>
    public bool WarnOnExpensive { get; set; } = true;

    /// <summary>
    /// Threshold for considering a computation as expensive (default: 10,000).
    /// Warns if Jacobian dimensions exceed this threshold.
    /// </summary>
    public int ExpensiveThreshold { get; set; } = 10000;

    /// <summary>
    /// Options passed to VJP computation when using reverse mode.
    /// </summary>
    public VectorJacobianProduct.VJPOptions? VJPOptions { get; set; } = null;

    /// <summary>
    /// Options passed to JVP computation when using forward mode.
    /// </summary>
    public JacobianVectorProduct.JVPOptions? JVPOptions { get; set; } = null;
}

/// <summary>
/// Result of Jacobian computation including detected structure.
/// </summary>
public class JacobianResult
{
    /// <summary>
    /// The computed Jacobian matrix tensor.
    /// </summary>
    public Tensor Jacobian { get; set; } = null!;

    /// <summary>
    /// The detected structure type of the Jacobian.
    /// </summary>
    public JacobianStructure Structure { get; set; } = JacobianStructure.General;

    /// <summary>
    /// The computation mode used (actual mode, may differ from requested if Auto was specified).
    /// </summary>
    public JacobianMode ModeUsed { get; set; } = JacobianMode.Auto;

    /// <summary>
    /// Whether the Jacobian is sparse (contains mostly zeros).
    /// </summary>
    public bool IsSparse { get; set; } = false;

    /// <summary>
    /// Sparsity ratio (fraction of zero entries, if sparse).
    /// </summary>
    public float SparsityRatio { get; set; } = 0.0f;

    /// <summary>
    /// Number of non-zero entries in the Jacobian.
    /// </summary>
    public int NonZeroCount { get; set; } = 0;
}

/// <summary>
/// Static class for computing full Jacobian matrices of vector-valued functions.
/// The implementation automatically selects the most efficient strategy (VJP or JVP)
/// based on input and output dimensions, and supports sparse representations.
/// </summary>
public static class Jacobian
{
    /// <summary>
    /// Computes the full Jacobian matrix for a function f: R^n → R^m.
    /// Automatically selects the most efficient computation mode.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a tensor.</param>
    /// <param name="x">The input tensor at which to compute the Jacobian.</param>
    /// <returns>The Jacobian matrix as a tensor with shape [m, n].</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor Compute(Func<Tensor, Tensor> f, Tensor x)
    {
        var result = ComputeWithOptions(f, x, new JacobianOptions());
        return result.Jacobian;
    }

    /// <summary>
    /// Computes the full Jacobian matrix with custom options.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the Jacobian.</param>
    /// <param name="options">Options for Jacobian computation.</param>
    /// <returns>The Jacobian matrix as a tensor with shape [m, n].</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or options is null.</exception>
    public static Tensor Compute(Func<Tensor, Tensor> f, Tensor x, JacobianOptions options)
    {
        var result = ComputeWithOptions(f, x, options);
        return result.Jacobian;
    }

    /// <summary>
    /// Computes the full Jacobian matrix and returns detailed result information.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the Jacobian.</param>
    /// <param name="options">Options for Jacobian computation.</param>
    /// <returns>JacobianResult containing the Jacobian matrix and metadata.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or options is null.</exception>
    public static JacobianResult ComputeWithOptions(Func<Tensor, Tensor> f, Tensor x, JacobianOptions options)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (options == null)
            throw new ArgumentNullException(nameof(options));

        // Forward pass to determine dimensions
        var y = f(TensorAccessor.CloneWithoutGrad(x));
        var n = x.Size; // Input dimension
        var m = y.Size; // Output dimension

        // Warn about potentially expensive computations
        if (options.WarnOnExpensive && m * n > options.ExpensiveThreshold)
        {
            Console.WriteLine($"Warning: Computing {m}x{n} Jacobian may be expensive. Consider using sparse mode or specifying output indices.");
        }

        // Determine actual computation mode
        var modeUsed = DetermineMode(options.Mode, m, n);

        // Select output indices if partial Jacobian requested
        var outputIndices = options.OutputIndices;
        var effectiveM = outputIndices != null ? outputIndices.Length : m;

        // Initialize result
        var result = new JacobianResult
        {
            ModeUsed = modeUsed,
            Structure = JacobianStructure.General
        };

        // Compute Jacobian using selected strategy
        Tensor jacobian;
        if (modeUsed == JacobianMode.Reverse)
        {
            // Column-by-column using VJP
            jacobian = ComputeWithVJP(f, x, y.Shape, effectiveM, n, outputIndices, options);
        }
        else // Forward mode
        {
            // Row-by-row using JVP
            jacobian = ComputeWithJVP(f, x, effectiveM, n, outputIndices, options);
        }

        // Detect structure if requested
        if (options.DetectStructure)
        {
            DetectJacobianStructure(jacobian, ref result);
        }

        // Compute sparsity statistics
        ComputeSparsityStatistics(jacobian, ref result, options.SparsityThreshold);

        // Apply sparse representation if requested
        if (options.Sparse)
        {
            jacobian = CreateSparseTensor(jacobian, options.SparsityThreshold);
        }

        result.Jacobian = jacobian;
        return result;
    }

    /// <summary>
    /// Computes Jacobian column-by-column using Vector-Jacobian Products (VJP).
    /// Each column of the Jacobian is computed as J[:,j] = VJP(f, x, e_j) where e_j is the j-th standard basis vector.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <param name="m">The number of output dimensions.</param>
    /// <param name="n">The number of input dimensions.</param>
    /// <param name="outputIndices">Optional indices for partial Jacobian.</param>
    /// <param name="options">Options for computation.</param>
    /// <returns>The Jacobian matrix.</returns>
    private static Tensor ComputeWithVJP(
        Func<Tensor, Tensor> f,
        Tensor x,
        int[] outputShape,
        int m,
        int n,
        int[]? outputIndices,
        JacobianOptions options)
    {
        var jacobianData = new float[m * n];

        // Create VJP options
        var vjpOptions = options.VJPOptions ?? new VectorJacobianProduct.VJPOptions
        {
            EnableParallel = options.EnableParallel,
            MaxParallelTasks = options.MaxParallelTasks
        };

        if (options.EnableParallel && n > 1)
        {
            // Parallel computation for columns
            var maxDegreeOfParallelism = options.MaxParallelTasks > 0
                ? options.MaxParallelTasks
                : Environment.ProcessorCount;

            Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, j =>
            {
                // Create standard basis vector e_j
                var e_j = CreateStandardBasisVector(outputShape, j, outputIndices);
                var column = VectorJacobianProduct.Compute(f, x, e_j, vjpOptions);
                var columnData = TensorAccessor.GetData(column);

                // Copy column to Jacobian
                for (int i = 0; i < m; i++)
                {
                    jacobianData[i * n + j] = columnData[i];
                }

                // Report progress
                options.ProgressCallback?.Invoke(j + 1, n);
            });
        }
        else
        {
            // Sequential computation
            for (int j = 0; j < n; j++)
            {
                // Create standard basis vector e_j
                var e_j = CreateStandardBasisVector(outputShape, j, outputIndices);
                var column = VectorJacobianProduct.Compute(f, x, e_j, vjpOptions);
                var columnData = TensorAccessor.GetData(column);

                // Copy column to Jacobian
                for (int i = 0; i < m; i++)
                {
                    jacobianData[i * n + j] = columnData[i];
                }

                // Report progress
                options.ProgressCallback?.Invoke(j + 1, n);
            }
        }

        return new Tensor(jacobianData, new[] { m, n });
    }

    /// <summary>
    /// Computes Jacobian row-by-row using Jacobian-Vector Products (JVP).
    /// Each row of the Jacobian is computed as J[i,:] = JVP(f, x, e_i) where e_i is the i-th standard basis vector.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor.</param>
    /// <param name="m">The number of output dimensions.</param>
    /// <param name="n">The number of input dimensions.</param>
    /// <param name="outputIndices">Optional indices for partial Jacobian.</param>
    /// <param name="options">Options for computation.</param>
    /// <returns>The Jacobian matrix.</returns>
    private static Tensor ComputeWithJVP(
        Func<Tensor, Tensor> f,
        Tensor x,
        int m,
        int n,
        int[]? outputIndices,
        JacobianOptions options)
    {
        var jacobianData = new float[m * n];

        // Create JVP options
        var jvpOptions = options.JVPOptions ?? new JacobianVectorProduct.JVPOptions
        {
            EnableParallel = options.EnableParallel,
            MaxParallelTasks = options.MaxParallelTasks
        };

        if (options.EnableParallel && m > 1)
        {
            // Parallel computation for rows
            var maxDegreeOfParallelism = options.MaxParallelTasks > 0
                ? options.MaxParallelTasks
                : Environment.ProcessorCount;

            Parallel.For(0, m, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, i =>
            {
                // Create standard basis vector e_i
                var e_i = CreateStandardBasisVector(x.Shape, i, null);
                var row = JacobianVectorProduct.Compute(f, x, e_i, jvpOptions);
                var rowData = TensorAccessor.GetData(row);

                // Copy row to Jacobian
                for (int j = 0; j < n; j++)
                {
                    jacobianData[i * n + j] = rowData[j];
                }

                // Report progress
                options.ProgressCallback?.Invoke(i + 1, m);
            });
        }
        else
        {
            // Sequential computation
            for (int i = 0; i < m; i++)
            {
                // Create standard basis vector e_i
                var e_i = CreateStandardBasisVector(x.Shape, i, null);
                var row = JacobianVectorProduct.Compute(f, x, e_i, jvpOptions);
                var rowData = TensorAccessor.GetData(row);

                // Copy row to Jacobian
                for (int j = 0; j < n; j++)
                {
                    jacobianData[i * n + j] = rowData[j];
                }

                // Report progress
                options.ProgressCallback?.Invoke(i + 1, m);
            }
        }

        return new Tensor(jacobianData, new[] { m, n });
    }

    /// <summary>
    /// Creates a standard basis vector with 1 at the specified index.
    /// </summary>
    /// <param name="shape">The shape of the vector.</param>
    /// <param name="index">The index where to place the 1.</param>
    /// <param name="outputIndices">Optional indices for partial selection.</param>
    /// <returns>A standard basis vector tensor.</returns>
    private static Tensor CreateStandardBasisVector(int[] shape, int index, int[]? outputIndices)
    {
        var vector = Tensor.Zeros(shape);
        var data = TensorAccessor.GetData(vector);

        if (outputIndices != null && index < outputIndices.Length)
        {
            // For partial Jacobian, place 1 at the actual output index
            data[outputIndices[index]] = 1.0f;
        }
        else
        {
            // For full Jacobian, place 1 at the specified index
            data[index] = 1.0f;
        }

        return vector;
    }

    /// <summary>
    /// Determines the most efficient computation mode based on dimensions.
    /// </summary>
    /// <param name="requestedMode">The mode requested by the user.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="n">Input dimension.</param>
    /// <returns>The actual mode to use.</returns>
    private static JacobianMode DetermineMode(JacobianMode requestedMode, int m, int n)
    {
        if (requestedMode != JacobianMode.Auto)
        {
            return requestedMode;
        }

        // Auto mode: choose based on which dimension is smaller
        // VJP (reverse) is O(m) backward passes - better when m << n
        // JVP (forward) is O(n) forward passes - better when n << m
        return m <= n ? JacobianMode.Reverse : JacobianMode.Forward;
    }

    /// <summary>
    /// Detects special structures in the Jacobian matrix.
    /// </summary>
    /// <param name="jacobian">The Jacobian matrix.</param>
    /// <param name="result">The result object to update with structure information.</param>
    private static void DetectJacobianStructure(Tensor jacobian, ref JacobianResult result)
    {
        var data = TensorAccessor.GetData(jacobian);
        var shape = jacobian.Shape;

        if (shape.Length != 2 || shape[0] != shape[1])
        {
            // Only check diagonal structure for square matrices
            return;
        }

        var n = shape[0];
        bool isDiagonal = true;

        // Check for diagonal structure
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != j && Math.Abs(data[i * n + j]) > 1e-10f)
                {
                    isDiagonal = false;
                    break;
                }
            }
            if (!isDiagonal) break;
        }

        if (isDiagonal)
        {
            result.Structure = JacobianStructure.Diagonal;
            return;
        }

        // Check for block-diagonal structure (simplified check for 2x2 blocks)
        // This is a basic implementation - a full implementation would use graph algorithms
        var isBlockDiagonal = true;
        int blockSize = 2;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int blockI = i / blockSize;
                int blockJ = j / blockSize;
                if (blockI != blockJ && Math.Abs(data[i * n + j]) > 1e-10f)
                {
                    isBlockDiagonal = false;
                    break;
                }
            }
            if (!isBlockDiagonal) break;
        }

        if (isBlockDiagonal)
        {
            result.Structure = JacobianStructure.BlockDiagonal;
            return;
        }

        // Check for triangular structure
        bool isUpperTriangular = true;
        bool isLowerTriangular = true;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i > j && Math.Abs(data[i * n + j]) > 1e-10f)
                {
                    isUpperTriangular = false;
                }
                if (i < j && Math.Abs(data[i * n + j]) > 1e-10f)
                {
                    isLowerTriangular = false;
                }
            }
        }

        if (isUpperTriangular || isLowerTriangular)
        {
            result.Structure = JacobianStructure.Triangular;
        }
    }

    /// <summary>
    /// Computes sparsity statistics for the Jacobian.
    /// </summary>
    /// <param name="jacobian">The Jacobian matrix.</param>
    /// <param name="result">The result object to update with sparsity information.</param>
    /// <param name="threshold">The threshold for considering a value as zero.</param>
    private static void ComputeSparsityStatistics(Tensor jacobian, ref JacobianResult result, float threshold)
    {
        var data = TensorAccessor.GetData(jacobian);
        int nonZeroCount = 0;

        for (int i = 0; i < data.Length; i++)
        {
            if (Math.Abs(data[i]) > threshold)
            {
                nonZeroCount++;
            }
        }

        result.NonZeroCount = nonZeroCount;
        result.SparsityRatio = 1.0f - ((float)nonZeroCount / data.Length);
        result.IsSparse = result.SparsityRatio > 0.9f; // Consider sparse if > 90% zeros
    }

    /// <summary>
    /// Creates a sparse tensor representation of the Jacobian.
    /// This is a placeholder - a full implementation would use a proper sparse tensor class.
    /// </summary>
    /// <param name="denseJacobian">The dense Jacobian tensor.</param>
    /// <param name="threshold">The threshold for considering a value as zero.</param>
    /// <returns>A sparse representation (currently returns dense tensor).</returns>
    private static Tensor CreateSparseTensor(Tensor denseJacobian, float threshold)
    {
        // For now, just return the dense tensor
        // A full implementation would create a SparseTensor class with CSR/CSC format
        return denseJacobian;
    }

    /// <summary>
    /// Computes a numerical approximation of the Jacobian using finite differences.
    /// Useful for validation and testing.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the Jacobian.</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>The numerical approximation of the Jacobian.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor ComputeNumerical(Func<Tensor, Tensor> f, Tensor x, float epsilon = 1e-5f)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var y0 = f(x);
        var outputSize = y0.Size;
        var inputSize = x.Size;

        var jacobianData = new float[outputSize * inputSize];
        var xData = TensorAccessor.GetData(x);

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

            // Compute partial derivatives
            for (int i = 0; i < outputSize; i++)
            {
                jacobianData[i * inputSize + j] = (yPlusData[i] - y0Data[i]) / epsilon;
            }
        }

        return new Tensor(jacobianData, new[] { outputSize, inputSize });
    }

    /// <summary>
    /// Validates Jacobian computation against numerical approximation.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor at which to compute the Jacobian.</param>
    /// <param name="jacobian">The computed Jacobian to validate.</param>
    /// <param name="tolerance">The tolerance for comparison (default: 1e-6).</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>True if Jacobian matches numerical approximation within tolerance, false otherwise.</returns>
    public static bool Validate(
        Func<Tensor, Tensor> f,
        Tensor x,
        Tensor jacobian,
        float tolerance = 1e-6f,
        float epsilon = 1e-5f)
    {
        var jacobianNumerical = ComputeNumerical(f, x, epsilon);

        var jacobianData = TensorAccessor.GetData(jacobian);
        var jacobianNumericalData = TensorAccessor.GetData(jacobianNumerical);

        if (jacobianData.Length != jacobianNumericalData.Length)
        {
            return false;
        }

        for (int i = 0; i < jacobianData.Length; i++)
        {
            if (Math.Abs(jacobianData[i] - jacobianNumericalData[i]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Computes the Vector-Jacobian Product (VJP) for efficiency.
    /// This is used when you only need the product of a vector with the Jacobian.
    /// This method provides backward compatibility with the old API.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a tensor.</param>
    /// <param name="x">The input tensor at which to compute the VJP.</param>
    /// <param name="v">The vector to multiply with the Jacobian (must match output size).</param>
    /// <returns>The vector-Jacobian product v^T * J.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or v is null.</exception>
    /// <exception cref="ArgumentException">Thrown when v shape doesn't match output shape.</exception>
    public static Tensor ComputeVectorJacobianProduct(Func<Tensor, Tensor> f, Tensor x, Tensor v)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (v == null)
            throw new ArgumentNullException(nameof(v));

        var xGrad = TensorAccessor.CloneWithGrad(x);
        var y = f(xGrad);

        if (!y.Shape.SequenceEqual(v.Shape))
            throw new ArgumentException("Vector v must match output shape of function f");

        // Backward pass with custom gradient output (v)
        y.Backward(v);

        return xGrad.Gradient!;
    }
}
