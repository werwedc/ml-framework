using System;

namespace MLFramework.HAL.CUDA.Graphs;

/// <summary>
/// Handles fallback logic for CUDA graph capture failures
/// </summary>
public class CUDAGraphFallbackHandler : IDisposable
{
    private readonly CUDAGraphFallbackStrategy _strategy;
    private readonly int _maxRetries;
    private int _captureAttempts;
    private bool _useFallback;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the CUDAGraphFallbackHandler class
    /// </summary>
    /// <param name="strategy">Fallback strategy to use</param>
    /// <param name="maxRetries">Maximum number of retries for RetryThenFallback strategy</param>
    public CUDAGraphFallbackHandler(
        CUDAGraphFallbackStrategy strategy = CUDAGraphFallbackStrategy.CaptureOrFallback,
        int maxRetries = 3)
    {
        _strategy = strategy;
        _maxRetries = maxRetries;
        _captureAttempts = 0;
        _useFallback = false;
        _disposed = false;
    }

    /// <summary>
    /// Gets whether to use fallback for the next execution
    /// </summary>
    public bool ShouldUseFallback =>
        _strategy == CUDAGraphFallbackStrategy.NeverCapture || _useFallback;

    /// <summary>
    /// Gets the number of capture attempts made
    /// </summary>
    public int CaptureAttempts => _captureAttempts;

    /// <summary>
    /// Gets the current fallback strategy
    /// </summary>
    public CUDAGraphFallbackStrategy Strategy => _strategy;

    /// <summary>
    /// Tries to execute with graph capture, falls back on failure
    /// </summary>
    /// <typeparam name="TReturn">Return type</typeparam>
    /// <param name="captureFunc">Function to capture the graph</param>
    /// <param name="regularFunc">Regular execution function as fallback</param>
    /// <param name="stream">CUDA stream for execution</param>
    /// <returns>Default value of TReturn (actual work happens via side effects)</returns>
    public TReturn ExecuteWithFallback<TReturn>(
        Func<ICUDAGraph> captureFunc,
        Action<CudaStream> regularFunc,
        CudaStream stream)
    {
        ThrowIfDisposed();

        if (ShouldUseFallback)
        {
            // Use regular execution
            regularFunc(stream);
            return default!;
        }

        // Try to capture and execute
        try
        {
            var graph = captureFunc();
            _captureAttempts++;
            graph.Execute(stream);

            // Success - check if we need to update state
            UpdateStateAfterSuccess();

            return default!;
        }
        catch (Exception ex)
        {
            // Handle failure based on strategy
            return HandleFailure<TReturn>(ex, regularFunc, stream);
        }
    }

    /// <summary>
    /// Tries to execute with graph capture
    /// </summary>
    /// <param name="captureFunc">Function to capture the graph</param>
    /// <param name="regularFunc">Regular execution function as fallback</param>
    /// <param name="stream">CUDA stream for execution</param>
    /// <param name="usedGraph">Output parameter indicating whether graph was used</param>
    public void TryExecuteWithFallback(
        Func<ICUDAGraph> captureFunc,
        Action<CudaStream> regularFunc,
        CudaStream stream,
        out bool usedGraph)
    {
        ThrowIfDisposed();

        usedGraph = false;

        if (ShouldUseFallback)
        {
            // Use regular execution
            regularFunc(stream);
            return;
        }

        // Try to capture and execute
        try
        {
            var graph = captureFunc();
            _captureAttempts++;
            graph.Execute(stream);
            usedGraph = true;

            // Success - check if we need to update state
            UpdateStateAfterSuccess();
        }
        catch (Exception ex)
        {
            // Handle failure based on strategy
            HandleFailure(ex, regularFunc, stream);
        }
    }

    /// <summary>
    /// Handles capture failure based on strategy
    /// </summary>
    private TReturn HandleFailure<TReturn>(Exception ex, Action<CudaStream> regularFunc, CudaStream stream)
    {
        switch (_strategy)
        {
            case CUDAGraphFallbackStrategy.CaptureOnly:
                // Re-throw exception
                throw new CUDAGraphCaptureException("Graph capture failed", ex);

            case CUDAGraphFallbackStrategy.CaptureOrFallback:
            case CUDAGraphFallbackStrategy.TryOnceThenFallback:
                // Log warning and fallback
                LogFallbackWarning(ex);
                _useFallback = true;
                regularFunc(stream);
                break;

            case CUDAGraphFallbackStrategy.RetryThenFallback:
                // Retry if max retries not exceeded
                if (_captureAttempts < _maxRetries)
                {
                    LogRetryWarning(ex);
                    regularFunc(stream); // Fallback for this attempt
                }
                else
                {
                    // Max retries exceeded, permanently fallback
                    LogMaxRetriesWarning();
                    _useFallback = true;
                    regularFunc(stream);
                }
                break;

            case CUDAGraphFallbackStrategy.NeverCapture:
                // Should never reach here
                regularFunc(stream);
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(_strategy), $"Unknown strategy: {_strategy}");
        }

        return default!;
    }

    /// <summary>
    /// Handles capture failure based on strategy (void overload)
    /// </summary>
    private void HandleFailure(Exception ex, Action<CudaStream> regularFunc, CudaStream stream)
    {
        HandleFailure<object>(ex, regularFunc, stream);
    }

    /// <summary>
    /// Updates handler state after successful capture
    /// </summary>
    private void UpdateStateAfterSuccess()
    {
        // Reset fallback flag on success (for retry strategies)
        if (_strategy == CUDAGraphFallbackStrategy.RetryThenFallback)
        {
            _useFallback = false;
        }
    }

    /// <summary>
    /// Logs a warning when falling back to regular execution
    /// </summary>
    private void LogFallbackWarning(Exception ex)
    {
        Console.WriteLine($"[CUDA Graph Fallback] Falling back to regular execution: {ex.Message}");
    }

    /// <summary>
    /// Logs a warning when capture attempt fails but will be retried
    /// </summary>
    private void LogRetryWarning(Exception ex)
    {
        Console.WriteLine($"[CUDA Graph Retry] Capture attempt {_captureAttempts + 1} failed: {ex.Message}");
    }

    /// <summary>
    /// Logs a warning when max retries have been exceeded
    /// </summary>
    private void LogMaxRetriesWarning()
    {
        Console.WriteLine($"[CUDA Graph Fallback] Max retries ({_maxRetries}) exceeded, using fallback");
    }

    /// <summary>
    /// Resets the fallback handler state
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();
        _captureAttempts = 0;
        _useFallback = false;
    }

    /// <summary>
    /// Disposes the fallback handler
    /// </summary>
    public void Dispose()
    {
        _disposed = true;
    }

    /// <summary>
    /// Throws an exception if the handler has been disposed
    /// </summary>
    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CUDAGraphFallbackHandler));
    }
}
