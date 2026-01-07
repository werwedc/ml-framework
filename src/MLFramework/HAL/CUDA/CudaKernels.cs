using System.Runtime.InteropServices;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// P/Invoke declarations for CUDA kernels and operations
/// Note: The actual CUDA kernels will be implemented in CUDA C++ (.cu files)
/// and compiled into mlframework_cuda.dll
/// </summary>
public static class CudaKernels
{
    private const string CudaLibrary = "mlframework_cuda.dll";

    #region Element-wise Arithmetic Kernels

    /// <summary>
    /// Add two tensors element-wise
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaAdd(
        IntPtr result,
        IntPtr a,
        IntPtr b,
        long size);

    /// <summary>
    /// Subtract two tensors element-wise
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaSubtract(
        IntPtr result,
        IntPtr a,
        IntPtr b,
        long size);

    /// <summary>
    /// Multiply two tensors element-wise
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaMultiply(
        IntPtr result,
        IntPtr a,
        IntPtr b,
        long size);

    /// <summary>
    /// Divide two tensors element-wise
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaDivide(
        IntPtr result,
        IntPtr a,
        IntPtr b,
        long size);

    #endregion

    #region Activation Kernels

    /// <summary>
    /// ReLU activation: max(0, x)
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaReLU(
        IntPtr result,
        IntPtr input,
        long size);

    /// <summary>
    /// Sigmoid activation: 1 / (1 + exp(-x))
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaSigmoid(
        IntPtr result,
        IntPtr input,
        long size);

    /// <summary>
    /// Tanh activation
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaTanh(
        IntPtr result,
        IntPtr input,
        long size);

    #endregion

    #region Reduction Kernels

    /// <summary>
    /// Sum reduction: compute sum of all elements
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaSum(
        IntPtr result,
        IntPtr input,
        long size);

    /// <summary>
    /// Max reduction: compute maximum of all elements
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaMax(
        IntPtr result,
        IntPtr input,
        long size);

    /// <summary>
    /// Min reduction: compute minimum of all elements
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaMin(
        IntPtr result,
        IntPtr input,
        long size);

    #endregion

    #region Memory Operations

    /// <summary>
    /// Copy tensor data from source to destination
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaCopy(
        IntPtr result,
        IntPtr input,
        long size);

    /// <summary>
    /// Fill tensor with a scalar value
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void CudaFill(
        IntPtr result,
        float value,
        long size);

    #endregion
}
