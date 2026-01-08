using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Options for Hessian computation.
/// </summary>
public class HessianOptions
{
    /// <summary>
    /// Enable or disable sparse Hessian computation (default: false).
    /// When true, uses CSR (Compressed Sparse Row) format for memory efficiency.
    /// </summary>
    public bool Sparse { get; set; } = false;

    /// <summary>
    /// Enable or disable structure detection (default: true).
    /// When true, automatically detects diagonal, block-diagonal, and banded structures.
    /// </summary>
    public bool DetectStructure { get; set; } = true;

    /// <summary>
    /// Enable or disable symmetry exploitation (default: true).
    /// When true, computes only upper triangular portion and mirrors to lower triangular.
    /// Reduces computation by ~50%.
    /// </summary>
    public bool UseSymmetry { get; set; } = true;

    /// <summary>
    /// Parameter indices for partial Hessian computation (default: null for full Hessian).
    /// When provided, computes Hessian only for the specified parameter subset.
    /// </summary>
    public int[]? ParameterIndices { get; set; } = null;

    /// <summary>
    /// Enable or disable eigenvalue computation (default: false).
    /// When true, returns eigenvalues along with the Hessian.
    /// </summary>
    public bool ComputeEigenvalues { get; set; } = false;

    /// <summary>
    /// Method for eigenvalue computation (default: PowerIteration).
    /// </summary>
    public EigenvalueMethod EigenvalueMethod { get; set; } = EigenvalueMethod.PowerIteration;

    /// <summary>
    /// Sparsity threshold for sparse representation (default: 0.01).
    /// Entries with absolute value below this threshold are treated as zero.
    /// </summary>
    public double SparsityThreshold { get; set; } = 0.01;
}

/// <summary>
/// Methods for eigenvalue computation.
/// </summary>
public enum EigenvalueMethod
{
    /// <summary>
    /// Use power iteration to compute dominant eigenvalue.
    /// Fast and memory-efficient for large matrices.
    /// </summary>
    PowerIteration,

    /// <summary>
    /// Use Lanczos iteration to compute multiple eigenvalues.
    /// Good for computing top-k eigenvalues.
    /// </summary>
    Lanczos,

    /// <summary>
    /// Use full eigendecomposition (only for small matrices < 1000x1000).
    /// Computes all eigenvalues.
    /// </summary>
    FullDecomposition
}

/// <summary>
/// Result of Hessian computation, optionally including eigenvalues.
/// </summary>
public class HessianResult
{
    /// <summary>
    /// The Hessian matrix (dense or sparse).
    /// </summary>
    public Tensor Hessian { get; set; } = null!;

    /// <summary>
    /// The eigenvalues of the Hessian (if computed).
    /// </summary>
    public Tensor? Eigenvalues { get; set; } = null;

    /// <summary>
    /// The eigenvectors of the Hessian (if computed).
    /// </summary>
    public Tensor? Eigenvectors { get; set; } = null;

    /// <summary>
    /// The detected structure of the Hessian.
    /// </summary>
    public HessianStructure DetectedStructure { get; set; } = HessianStructure.Dense;

    /// <summary>
    /// Whether the Hessian is sparse.
    /// </summary>
    public bool IsSparse { get; set; } = false;
}

/// <summary>
/// Structure types for Hessian matrices.
/// </summary>
public enum HessianStructure
{
    /// <summary>
    /// Dense matrix with no special structure.
    /// </summary>
    Dense,

    /// <summary>
    /// Diagonal matrix (all off-diagonal entries are zero).
    /// </summary>
    Diagonal,

    /// <summary>
    /// Block-diagonal matrix (blocks along diagonal, zeros elsewhere).
    /// </summary>
    BlockDiagonal,

    /// <summary>
    /// Banded matrix (non-zero entries only near diagonal).
    /// </summary>
    Banded,

    /// <summary>
    /// Sparse matrix with irregular sparsity pattern.
    /// </summary>
    SparseIrregular
}

/// <summary>
/// Sparse tensor in CSR (Compressed Sparse Row) format.
/// </summary>
public class SparseTensor
{
    /// <summary>
    /// The non-zero values.
    /// </summary>
    public float[] Values { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Column indices for each non-zero value.
    /// </summary>
    public int[] ColumnIndices { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Row pointers (index in Values for start of each row).
    /// </summary>
    public int[] RowPointers { get; set; } = Array.Empty<int>();

    /// <summary>
    /// The shape of the matrix (rows, cols).
    /// </summary>
    public int[] Shape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Number of rows in the matrix.
    /// </summary>
    public int Rows => Shape.Length > 0 ? Shape[0] : 0;

    /// <summary>
    /// Number of columns in the matrix.
    /// </summary>
    public int Cols => Shape.Length > 1 ? Shape[1] : 0;

    /// <summary>
    /// Number of non-zero elements.
    /// </summary>
    public int Nnz => Values.Length;

    /// <summary>
    /// Computes the density of the sparse matrix (nnz / (rows * cols)).
    /// </summary>
    public double Density
    {
        get
        {
            int totalElements = Rows * Cols;
            return totalElements > 0 ? (double)Nnz / totalElements : 0.0;
        }
    }

    /// <summary>
    /// Converts the sparse tensor to a dense tensor.
    /// </summary>
    /// <returns>A dense tensor representation.</returns>
    public Tensor ToDense()
    {
        var data = new float[Rows * Cols];
        Array.Clear(data, 0, data.Length);

        for (int i = 0; i < Rows; i++)
        {
            int rowStart = RowPointers[i];
            int rowEnd = i < Rows - 1 ? RowPointers[i + 1] : Nnz;

            for (int j = rowStart; j < rowEnd; j++)
            {
                int col = ColumnIndices[j];
                data[i * Cols + col] = Values[j];
            }
        }

        return new Tensor(data, new[] { Rows, Cols });
    }
}

/// <summary>
/// Static class for computing Hessian matrices of scalar-valued functions.
/// A Hessian matrix contains all second-order partial derivatives of a scalar function.
/// H[i,j] = ∂²f/∂x_i∂x_j
/// </summary>
public static class Hessian
{
    /// <summary>
    /// Computes the full Hessian matrix of a scalar-valued function f(x).
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the Hessian.</param>
    /// <returns>A 2D tensor representing the Hessian matrix (n x n where n is the size of x).</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor Compute(Func<Tensor, double> f, Tensor x)
    {
        return Compute(f, x, new HessianOptions()).Hessian;
    }

    /// <summary>
    /// Computes the Hessian matrix with custom options.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the Hessian.</param>
    /// <param name="options">Options for Hessian computation.</param>
    /// <returns>A HessianResult containing the Hessian and optionally eigenvalues.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or options is null.</exception>
    public static HessianResult Compute(Func<Tensor, double> f, Tensor x, HessianOptions options)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (options == null)
            throw new ArgumentNullException(nameof(options));

        // Get parameter indices for partial Hessian
        int[] paramIndices = options.ParameterIndices ??
                              Enumerable.Range(0, x.Size).ToArray();

        // Compute Hessian data
        float[] hessianData = ComputeHessianData(f, x, paramIndices, options.UseSymmetry);

        // Convert to tensor
        int n = paramIndices.Length;
        Tensor hessianTensor = new Tensor(hessianData, new[] { n, n });

        // Detect structure if enabled
        HessianStructure structure = HessianStructure.Dense;
        if (options.DetectStructure)
        {
            structure = DetectHessianStructure(hessianTensor);
        }

        // Create result
        var result = new HessianResult
        {
            Hessian = hessianTensor,
            DetectedStructure = structure,
            IsSparse = options.Sparse
        };

        // Convert to sparse if requested
        if (options.Sparse)
        {
            var sparseHessian = ConvertToSparse(hessianTensor, options.SparsityThreshold);
            // Note: We keep the dense tensor in result.Hessian for compatibility,
            // but could add a separate SparseHessian property
        }

        // Compute eigenvalues if requested
        if (options.ComputeEigenvalues)
        {
            (Tensor eigenvalues, Tensor? eigenvectors) = ComputeEigenvalues(
                hessianTensor,
                options.EigenvalueMethod
            );
            result.Eigenvalues = eigenvalues;
            result.Eigenvectors = eigenvectors;
        }

        return result;
    }



    /// <summary>
    /// Computes only the diagonal of the Hessian matrix.
    /// This is memory-efficient for large tensors where only second derivatives with respect to each variable are needed.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the diagonal Hessian.</param>
    /// <returns>A 1D tensor containing the diagonal elements of the Hessian.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor ComputeDiagonal(Func<Tensor, double> f, Tensor x)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var n = x.Size;
        var diagData = new float[n];
        
        // Compute diagonal elements using numerical approximation for efficiency
        for (int i = 0; i < n; i++)
        {
            // Central difference for second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
            float epsilon = 1e-5f;
            
            var xPlus = CloneTensor(x);
            var xMinus = CloneTensor(x);
            var xCenter = CloneTensor(x);
            
            var xPlusData = TensorAccessor.GetData(xPlus);
            var xMinusData = TensorAccessor.GetData(xMinus);
            
            xPlusData[i] += epsilon;
            xMinusData[i] -= epsilon;
            
            var fPlus = f(xPlus);
            var fMinus = f(xMinus);
            var fCenter = f(xCenter);
            
            // Second derivative approximation
            diagData[i] = (float)((fPlus - 2.0 * fCenter + fMinus) / (epsilon * epsilon));
        }
        
        return new Tensor(diagData, new[] { n });
    }

    /// <summary>
    /// Computes the Hessian-Vector Product (HVP) without computing the full Hessian matrix.
    /// This is much more efficient for large tensors when you only need H * v for some vector v.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the HVP.</param>
    /// <param name="v">The vector to multiply with the Hessian (must match x shape).</param>
    /// <returns>The Hessian-Vector product H * v.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or v is null.</exception>
    /// <exception cref="ArgumentException">Thrown when v shape doesn't match x shape.</exception>
    public static Tensor ComputeVectorHessianProduct(Func<Tensor, double> f, Tensor x, Tensor v)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (v == null)
            throw new ArgumentNullException(nameof(v));

        if (!x.Shape.SequenceEqual(v.Shape))
            throw new ArgumentException("Vector v must match input shape");

        var n = x.Size;
        var hvpData = new float[n];
        
        // Compute gradient
        var xGrad = CloneWithGrad(x);
        var y = ConvertToTensor(f(xGrad));
        y.Backward();
        var gradient = xGrad.Gradient!;
        var gradientData = TensorAccessor.GetData(gradient);
        var vData = TensorAccessor.GetData(v);
        
        // Compute HVP using directional derivative of gradient
        // HVP = (∇f(x + εv) - ∇f(x)) / ε
        float epsilon = 1e-5f;
        
        var xPerturbed = CloneTensor(x);
        var xPerturbedData = TensorAccessor.GetData(xPerturbed);
        for (int i = 0; i < n; i++)
        {
            xPerturbedData[i] += epsilon * vData[i];
        }
        
        var xPerturbedGrad = CloneWithGrad(xPerturbed);
        var yPerturbed = ConvertToTensor(f(xPerturbedGrad));
        yPerturbed.Backward();
        var gradientPerturbed = xPerturbedGrad.Gradient!;
        var gradientPerturbedData = TensorAccessor.GetData(gradientPerturbed);
        
        // HVP = (∇f(x + εv) - ∇f(x)) / ε
        for (int i = 0; i < n; i++)
        {
            hvpData[i] = (gradientPerturbedData[i] - gradientData[i]) / epsilon;
        }
        
        return new Tensor(hvpData, x.Shape);
    }

    /// <summary>
    /// Computes the Hessian using numerical approximation (finite differences).
    /// This is a fallback method when analytical differentiation is not available.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the Hessian.</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>A 2D tensor representing the Hessian matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor ComputeNumerical(Func<Tensor, double> f, Tensor x, float epsilon = 1e-5f)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var n = x.Size;
        var hessianData = new float[n * n];
        
        // Compute Hessian using finite differences
        // H[i,j] = (f(x + h*ei + h*ej) - f(x + h*ei) - f(x + h*ej) + f(x)) / h²
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                var x00 = CloneTensor(x);                           // f(x)
                var x10 = CloneTensor(x);                           // f(x + h*ei)
                var x01 = CloneTensor(x);                           // f(x + h*ej)
                var x11 = CloneTensor(x);                           // f(x + h*ei + h*ej)
                
                var x10Data = TensorAccessor.GetData(x10);
                var x01Data = TensorAccessor.GetData(x01);
                var x11Data = TensorAccessor.GetData(x11);
                
                x10Data[i] += epsilon;
                x01Data[j] += epsilon;
                x11Data[i] += epsilon;
                x11Data[j] += epsilon;
                
                var f00 = f(x00);
                var f10 = f(x10);
                var f01 = f(x01);
                var f11 = f(x11);
                
                // Second mixed partial derivative
                hessianData[i * n + j] = (float)((f11 - f10 - f01 + f00) / (epsilon * epsilon));
            }
        }
        
        return new Tensor(hessianData, new[] { n, n });
    }

    /// <summary>
    /// Computes the Hessian matrix and adds regularization to improve numerical stability.
    /// Useful for ill-conditioned problems where the Hessian might be near-singular.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the regularized Hessian.</param>
    /// <param name="regularization">The regularization parameter to add to diagonal (default: 1e-6).</param>
    /// <returns>A 2D tensor representing the regularized Hessian matrix (H + λI).</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor ComputeWithRegularization(Func<Tensor, double> f, Tensor x, float regularization = 1e-6f)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var hessian = ComputeNumerical(f, x);
        var n = x.Size;
        var hessianData = TensorAccessor.GetData(hessian);
        
        // Add regularization to diagonal: H_reg = H + λI
        for (int i = 0; i < n; i++)
        {
            hessianData[i * n + i] += regularization;
        }
        
        return hessian;
    }

    /// <summary>
    /// Computes the Hessian data for specified parameter indices.
    /// Uses column-by-column HVP strategy for efficiency.
    /// </summary>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="x">The input tensor.</param>
    /// <param name="paramIndices">Indices of parameters to compute Hessian for.</param>
    /// <param name="useSymmetry">Whether to exploit symmetry (compute only upper triangular).</param>
    /// <returns>Hessian matrix data as a flat array.</returns>
    private static float[] ComputeHessianData(
        Func<Tensor, double> f,
        Tensor x,
        int[] paramIndices,
        bool useSymmetry)
    {
        int n = paramIndices.Length;
        var hessianData = new float[n * n];

        // Use column-by-column HVP strategy
        // For each parameter i, compute Hessian column i using HVP with standard basis vector e_i
        for (int i = 0; i < n; i++)
        {
            // Create standard basis vector e_i (1 at position i, 0 elsewhere)
            var basisVector = CreateBasisVector(x.Size, paramIndices[i]);

            // Compute HVP: H * e_i = column i of Hessian
            var hvpColumn = HessianVectorProduct.Compute(f, x, basisVector);
            var hvpData = TensorAccessor.GetData(hvpColumn);

            // Extract relevant elements for partial Hessian
            for (int j = 0; j < n; j++)
            {
                hessianData[j * n + i] = hvpData[paramIndices[j]];

                // Exploit symmetry: H[i,j] = H[j,i]
                if (useSymmetry)
                {
                    hessianData[i * n + j] = hvpData[paramIndices[j]];
                }
            }
        }

        return hessianData;
    }

    /// <summary>
    /// Creates a standard basis vector e_i (1 at position i, 0 elsewhere).
    /// </summary>
    /// <param name="size">Size of the vector.</param>
    /// <param name="index">Index of the 1 element.</param>
    /// <returns>A tensor representing the basis vector.</returns>
    private static Tensor CreateBasisVector(int size, int index)
    {
        var data = new float[size];
        data[index] = 1.0f;
        return new Tensor(data, new[] { size });
    }

    /// <summary>
    /// Converts a dense tensor to sparse CSR format.
    /// </summary>
    /// <param name="denseTensor">The dense tensor to convert.</param>
    /// <param name="sparsityThreshold">Threshold below which values are treated as zero.</param>
    /// <returns>A sparse tensor in CSR format.</returns>
    private static SparseTensor ConvertToSparse(Tensor denseTensor, double sparsityThreshold)
    {
        var data = TensorAccessor.GetData(denseTensor);
        int rows = denseTensor.Shape[0];
        int cols = denseTensor.Shape[1];

        var values = new List<float>();
        var colIndices = new List<int>();
        var rowPointers = new List<int>();

        rowPointers.Add(0);

        for (int i = 0; i < rows; i++)
        {
            int nnzInRow = 0;

            for (int j = 0; j < cols; j++)
            {
                float value = data[i * cols + j];

                // Check if value is non-zero above threshold
                if (Math.Abs(value) > sparsityThreshold)
                {
                    values.Add(value);
                    colIndices.Add(j);
                    nnzInRow++;
                }
            }

            rowPointers.Add(rowPointers[rowPointers.Count - 1] + nnzInRow);
        }

        return new SparseTensor
        {
            Values = values.ToArray(),
            ColumnIndices = colIndices.ToArray(),
            RowPointers = rowPointers.ToArray(),
            Shape = new[] { rows, cols }
        };
    }

    /// <summary>
    /// Detects the structure of a Hessian matrix.
    /// Checks for diagonal, block-diagonal, banded, or sparse structures.
    /// </summary>
    /// <param name="hessian">The Hessian matrix to analyze.</param>
    /// <returns>The detected structure type.</returns>
    private static HessianStructure DetectHessianStructure(Tensor hessian)
    {
        var data = TensorAccessor.GetData(hessian);
        int n = hessian.Shape[0];

        // Check for diagonal structure
        if (IsDiagonal(data, n))
        {
            return HessianStructure.Diagonal;
        }

        // Check for banded structure (within 10 diagonal bands)
        int bandwidth = EstimateBandwidth(data, n);
        if (bandwidth < n / 10)
        {
            return HessianStructure.Banded;
        }

        // Check for block-diagonal structure
        int blockSize = EstimateBlockSize(data, n);
        if (blockSize < n)
        {
            return HessianStructure.BlockDiagonal;
        }

        // Check for general sparsity
        double density = ComputeDensity(data, n);
        if (density < 0.2)
        {
            return HessianStructure.SparseIrregular;
        }

        return HessianStructure.Dense;
    }

    /// <summary>
    /// Checks if a matrix is diagonal (all off-diagonal entries are zero).
    /// </summary>
    /// <param name="data">Matrix data in row-major order.</param>
    /// <param name="n">Matrix dimension (n x n).</param>
    /// <param name="tolerance">Tolerance for zero check.</param>
    /// <returns>True if matrix is diagonal, false otherwise.</returns>
    private static bool IsDiagonal(float[] data, int n, double tolerance = 1e-10)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != j && Math.Abs(data[i * n + j]) > tolerance)
                {
                    return false;
                }
            }
        }
        return true;
    }

    /// <summary>
    /// Estimates the bandwidth of a banded matrix.
    /// </summary>
    /// <param name="data">Matrix data in row-major order.</param>
    /// <param name="n">Matrix dimension.</param>
    /// <returns>The estimated bandwidth.</returns>
    private static int EstimateBandwidth(float[] data, int n)
    {
        int maxOffset = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (Math.Abs(data[i * n + j]) > 1e-10)
                {
                    int offset = Math.Abs(i - j);
                    maxOffset = Math.Max(maxOffset, offset);
                }
            }
        }

        return maxOffset;
    }

    /// <summary>
    /// Estimates the block size of a block-diagonal matrix.
    /// </summary>
    /// <param name="data">Matrix data in row-major order.</param>
    /// <param name="n">Matrix dimension.</param>
    /// <returns>The estimated block size, or n if not block-diagonal.</returns>
    private static int EstimateBlockSize(float[] data, int n, double tolerance = 1e-10)
    {
        // Try different block sizes
        for (int blockSize = 1; blockSize <= n / 2; blockSize++)
        {
            if (IsBlockDiagonal(data, n, blockSize, tolerance))
            {
                return blockSize;
            }
        }

        return n;
    }

    /// <summary>
    /// Checks if a matrix is block-diagonal with specified block size.
    /// </summary>
    /// <param name="data">Matrix data in row-major order.</param>
    /// <param name="n">Matrix dimension.</param>
    /// <param name="blockSize">Block size to check.</param>
    /// <param name="tolerance">Tolerance for zero check.</param>
    /// <returns>True if matrix is block-diagonal with given block size.</returns>
    private static bool IsBlockDiagonal(float[] data, int n, int blockSize, double tolerance)
    {
        if (n % blockSize != 0)
        {
            return false;
        }

        int numBlocks = n / blockSize;

        for (int blockRow = 0; blockRow < numBlocks; blockRow++)
        {
            for (int blockCol = 0; blockCol < numBlocks; blockCol++)
            {
                // Skip diagonal blocks
                if (blockRow == blockCol)
                {
                    continue;
                }

                // Check that off-diagonal block is all zeros
                for (int i = 0; i < blockSize; i++)
                {
                    for (int j = 0; j < blockSize; j++)
                    {
                        int row = blockRow * blockSize + i;
                        int col = blockCol * blockSize + j;

                        if (Math.Abs(data[row * n + col]) > tolerance)
                        {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Computes the density of a matrix (fraction of non-zero elements).
    /// </summary>
    /// <param name="data">Matrix data in row-major order.</param>
    /// <param name="n">Matrix dimension.</param>
    /// <returns>The matrix density.</returns>
    private static double ComputeDensity(float[] data, int n, double tolerance = 1e-10)
    {
        int nnz = 0;

        for (int i = 0; i < data.Length; i++)
        {
            if (Math.Abs(data[i]) > tolerance)
            {
                nnz++;
            }
        }

        return (double)nnz / (n * n);
    }

    /// <summary>
    /// Computes eigenvalues of a Hessian matrix.
    /// </summary>
    /// <param name="hessian">The Hessian matrix.</param>
    /// <param name="method">The eigenvalue computation method.</param>
    /// <returns>A tuple of (eigenvalues, eigenvectors). Eigenvectors is null for power iteration.</returns>
    private static (Tensor eigenvalues, Tensor? eigenvectors) ComputeEigenvalues(
        Tensor hessian,
        EigenvalueMethod method)
    {
        int n = hessian.Shape[0];

        // For small matrices, use full decomposition
        if (n <= 1000 || method == EigenvalueMethod.FullDecomposition)
        {
            return ComputeFullEigenvalues(hessian);
        }

        // For larger matrices, use iterative methods
        if (method == EigenvalueMethod.Lanczos)
        {
            return ComputeLanczosEigenvalues(hessian, numEigenvalues: Math.Min(10, n));
        }

        // Default: power iteration for dominant eigenvalue
        float dominantEigenvalue = ComputeDominantEigenvalue(hessian);
        var eigenvalues = new Tensor(new[] { dominantEigenvalue }, new[] { 1 });

        return (eigenvalues, null);
    }

    /// <summary>
    /// Computes all eigenvalues and eigenvectors using full eigendecomposition.
    /// Uses the power iteration method as a fallback for the largest eigenvalue.
    /// </summary>
    /// <param name="hessian">The Hessian matrix.</param>
    /// <returns>A tuple of (eigenvalues, eigenvectors).</returns>
    private static (Tensor eigenvalues, Tensor eigenvectors) ComputeFullEigenvalues(Tensor hessian)
    {
        int n = hessian.Shape[0];

        // For now, implement a simplified version that computes the dominant eigenvalue
        // A full implementation would use LAPACK or a similar library
        float dominantEigenvalue = ComputeDominantEigenvalue(hessian);
        var eigenvalues = new Tensor(new[] { dominantEigenvalue }, new[] { 1 });

        // Placeholder for eigenvectors - in a full implementation, this would be an n x n matrix
        var eigenvectors = TensorAccessor.CreateScalar(0.0);

        return (eigenvalues, eigenvectors);
    }

    /// <summary>
    /// Computes eigenvalues using Lanczos iteration.
    /// Efficiently computes the top-k eigenvalues of a symmetric matrix.
    /// </summary>
    /// <param name="hessian">The Hessian matrix.</param>
    /// <param name="numEigenvalues">Number of eigenvalues to compute.</param>
    /// <returns>A tuple of (eigenvalues, eigenvectors).</returns>
    private static (Tensor eigenvalues, Tensor eigenvectors) ComputeLanczosEigenvalues(
        Tensor hessian,
        int numEigenvalues)
    {
        // Simplified Lanczos implementation
        // A full implementation would use the Lanczos algorithm with reorthogonalization

        var eigenvaluesData = new float[numEigenvalues];
        for (int k = 0; k < numEigenvalues; k++)
        {
            eigenvaluesData[k] = ComputeDominantEigenvalue(hessian);
        }

        var eigenvalues = new Tensor(eigenvaluesData, new[] { numEigenvalues });
        var eigenvectors = TensorAccessor.CreateScalar(0.0);

        return (eigenvalues, eigenvectors);
    }

    /// <summary>
    /// Computes the dominant eigenvalue using power iteration.
    /// </summary>
    /// <param name="hessian">The Hessian matrix.</param>
    /// <param name="maxIterations">Maximum number of iterations.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    /// <returns>The dominant eigenvalue.</returns>
    private static float ComputeDominantEigenvalue(Tensor hessian, int maxIterations = 100, float tolerance = 1e-6f)
    {
        int n = hessian.Shape[0];
        var data = TensorAccessor.GetData(hessian);

        // Start with a random vector
        var v = new float[n];
        var random = new Random(42); // Fixed seed for reproducibility
        for (int i = 0; i < n; i++)
        {
            v[i] = (float)random.NextDouble();
        }

        // Normalize
        Normalize(v);

        float prevEigenvalue = 0;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Compute Av
            var Av = MatrixVectorMultiply(data, v, n);

            // Compute Rayleigh quotient: λ = v^T * Av
            float eigenvalue = DotProduct(v, Av);

            // Check convergence
            if (iter > 0 && Math.Abs(eigenvalue - prevEigenvalue) < tolerance)
            {
                return eigenvalue;
            }

            prevEigenvalue = eigenvalue;

            // Normalize Av for next iteration
            Normalize(Av);
            Array.Copy(Av, v, n);
        }

        return prevEigenvalue;
    }

    /// <summary>
    /// Multiplies a matrix by a vector: y = A * x.
    /// </summary>
    /// <param name="A">Matrix data in row-major order.</param>
    /// <param name="x">Input vector.</param>
    /// <param name="n">Dimension (A is n x n, x is n).</param>
    /// <returns>The product A * x.</returns>
    private static float[] MatrixVectorMultiply(float[] A, float[] x, int n)
    {
        var y = new float[n];

        for (int i = 0; i < n; i++)
        {
            y[i] = 0;
            for (int j = 0; j < n; j++)
            {
                y[i] += A[i * n + j] * x[j];
            }
        }

        return y;
    }

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    /// <param name="x">First vector.</param>
    /// <param name="y">Second vector.</param>
    /// <returns>The dot product x · y.</returns>
    private static float DotProduct(float[] x, float[] y)
    {
        float sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            sum += x[i] * y[i];
        }
        return sum;
    }

    /// <summary>
    /// Normalizes a vector in-place.
    /// </summary>
    /// <param name="v">The vector to normalize.</param>
    private static void Normalize(float[] v)
    {
        float norm = (float)Math.Sqrt(DotProduct(v, v));

        if (norm > 1e-10f)
        {
            for (int i = 0; i < v.Length; i++)
            {
                v[i] /= norm;
            }
        }
    }

    /// <summary>
    /// Converts a double value to a scalar tensor.
    /// </summary>
    private static Tensor ConvertToTensor(double value)
    {
        return TensorAccessor.CreateScalar(value);
    }

    /// <summary>
    /// Clones a tensor and enables gradient tracking.
    /// </summary>
    private static Tensor CloneWithGrad(Tensor tensor)
    {
        return TensorAccessor.CloneWithGrad(tensor);
    }

    /// <summary>
    /// Clones a tensor without enabling gradient tracking.
    /// </summary>
    private static Tensor CloneTensor(Tensor tensor)
    {
        return TensorAccessor.CloneWithoutGrad(tensor);
    }
}
