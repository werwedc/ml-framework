# Spec: Fallback Mechanism

## Overview
Implement a fallback mechanism that gracefully handles graph capture failures and unsupported operations by automatically falling back to individual kernel launches. This ensures robustness and prevents training failures.

## Requirements

### 1. CUDAGraphFallbackStrategy Enum
Define different fallback strategies.

```csharp
public enum CUDAGraphFallbackStrategy
{
    /// <summary>
    /// Always use regular execution, never capture
    /// </summary>
    NeverCapture,

    /// <summary>
    /// Capture if possible, fallback to regular execution on failure
    /// </summary>
    CaptureOrFallback,

    /// <summary>
    /// Capture only, throw exception on failure
    /// </summary>
    CaptureOnly,

    /// <summary>
    /// Try capture once, then permanently fallback on failure
    /// </summary>
    TryOnceThenFallback,

    /// <summary>
    /// Try capture multiple times before falling back
    /// </summary>
    RetryThenFallback
}
```

### 2. CUDAGraphFallbackHandler Class
Implement the fallback handler.

```csharp
public class CUDAGraphFallbackHandler : IDisposable
{
    private readonly CUDAGraphFallbackStrategy _strategy;
    private readonly int _maxRetries;
    private int _captureAttempts;
    private bool _useFallback;
    private bool _disposed;

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
    public bool ShouldUseFallback => _strategy == CUDAGraphFallbackStrategy.NeverCapture || _useFallback;

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
    public TReturn ExecuteWithFallback<TReturn>(
        Func<ICUDAGraph> captureFunc,
        Action<CUDAStream> regularFunc,
        CUDAStream stream)
    {
        if (ShouldUseFallback)
        {
            // Use regular execution
            regularFunc(stream);
            return default;
        }

        // Try to capture and execute
        try
        {
            var graph = captureFunc();
            _captureAttempts++;
            graph.Execute(stream);

            // Success - check if we need to update state
            UpdateStateAfterSuccess();

            return default;
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
    public void TryExecuteWithFallback(
        Func<ICUDAGraph> captureFunc,
        Action<CUDAStream> regularFunc,
        CUDAStream stream,
        out bool usedGraph)
    {
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

    private void HandleFailure<TReturn>(Exception ex, Action<CUDAStream> regularFunc, CUDAStream stream)
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
        }

        return default;
    }

    private void HandleFailure(Exception ex, Action<CUDAStream> regularFunc, CUDAStream stream)
    {
        HandleFailure<object>(ex, regularFunc, stream);
    }

    private void UpdateStateAfterSuccess()
    {
        // Reset fallback flag on success (for retry strategies)
        if (_strategy == CUDAGraphFallbackStrategy.RetryThenFallback)
        {
            _useFallback = false;
        }
    }

    private void LogFallbackWarning(Exception ex)
    {
        Console.WriteLine($"[CUDA Graph Fallback] Falling back to regular execution: {ex.Message}");
    }

    private void LogRetryWarning(Exception ex)
    {
        Console.WriteLine($"[CUDA Graph Retry] Capture attempt {_captureAttempts + 1} failed: {ex.Message}");
    }

    private void LogMaxRetriesWarning()
    {
        Console.WriteLine($"[CUDA Graph Fallback] Max retries ({_maxRetries}) exceeded, using fallback");
    }

    public void Reset()
    {
        _captureAttempts = 0;
        _useFallback = false;
    }

    public void Dispose()
    {
        _disposed = true;
    }
}
```

### 3. CUDAGraphCaptureException
Custom exception for graph capture failures.

```csharp
public class CUDAGraphCaptureException : Exception
{
    public CUDAGraphValidationResult ValidationResult { get; }

    public CUDAGraphCaptureException(string message)
        : base(message)
    {
        ValidationResult = null;
    }

    public CUDAGraphCaptureException(string message, Exception innerException)
        : base(message, innerException)
    {
        ValidationResult = null;
    }

    public CUDAGraphCaptureException(string message, CUDAGraphValidationResult validationResult)
        : base(message)
    {
        ValidationResult = validationResult;
    }

    public CUDAGraphCaptureException(
        string message,
        CUDAGraphValidationResult validationResult,
        Exception innerException)
        : base(message, innerException)
    {
        ValidationResult = validationResult;
    }
}
```

### 4. Extension Methods for Fallback
Provide convenient extension methods.

```csharp
public static class CUDAGraphFallbackExtensions
{
    /// <summary>
    /// Creates a fallback handler with the specified strategy
    /// </summary>
    public static CUDAGraphFallbackHandler WithFallback(
        this CUDAGraphManager manager,
        CUDAGraphFallbackStrategy strategy = CUDAGraphFallbackStrategy.CaptureOrFallback,
        int maxRetries = 3)
    {
        return new CUDAGraphFallbackHandler(strategy, maxRetries);
    }

    /// <summary>
    /// Executes with automatic fallback
    /// </summary>
    public static void ExecuteWithFallback(
        this CUDAGraphFallbackHandler handler,
        string graphName,
        Action<CUDAStream> captureAction,
        CUDAStream stream,
        CUDAGraphManager manager)
    {
        handler.TryExecuteWithFallback(
            () => manager.GetOrCaptureGraph(graphName, captureAction, stream),
            captureAction,
            stream,
            out _);
    }
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Graphs/Fallback/CUDAGraphFallbackStrategy.cs`
- **File**: `src/CUDA/Graphs/Fallback/CUDAGraphFallbackHandler.cs`
- **File**: `src/CUDA/Graphs/Fallback/CUDAGraphCaptureException.cs`
- **File**: `src/CUDA/Graphs/Fallback/CUDAGraphFallbackExtensions.cs`

### Dependencies
- ICUDAGraph interface (from spec_cuda_graph_core_interfaces)
- CUDAGraphValidationResult class (from spec_cuda_graph_core_interfaces)
- CUDAGraphManager (from spec_cuda_graph_manager)
- CUDAStream class (existing)
- System for Exception, Action, Func

### Fallback Strategy Behavior
1. **NeverCapture**: Always use regular execution
2. **CaptureOrFallback**: Try capture, fallback on failure
3. **CaptureOnly**: Capture only, throw on failure
4. **TryOnceThenFallback**: Try once, then permanently fallback
5. **RetryThenFallback**: Retry multiple times, then permanently fallback

### Error Handling
- Log warnings for fallbacks
- Provide detailed error messages
- Support retries for transient failures
- Clean up resources on fallback

## Success Criteria
- Fallback handler correctly determines when to use fallback
- Different strategies work as expected
- Errors are handled gracefully
- Retry logic works correctly
- Logging provides useful information
- Resources are cleaned up on failure
- Extension methods work as expected

## Testing Requirements

### Unit Tests
- Test NeverCapture strategy
- Test CaptureOrFallback strategy
- Test CaptureOnly strategy
- Test TryOnceThenFallback strategy
- Test RetryThenFallback strategy
- Test exception handling
- Test retry counting
- Test reset functionality

### Integration Tests
- Test fallback with actual graph capture failures (requires GPU)
- Test with unsupported operations
- Test with dynamic memory allocation
