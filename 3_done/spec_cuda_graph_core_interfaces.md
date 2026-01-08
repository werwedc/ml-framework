# Spec: CUDA Graph Core Interfaces

## Overview
Define the core interfaces and abstractions for CUDA Graph functionality. This spec creates the foundational types that all other CUDA Graph components will depend on.

## Requirements

### 1. ICUDAGraph Interface
Define the primary interface for CUDA graph operations.

```csharp
public interface ICUDAGraph : IDisposable
{
    /// <summary>
    /// Gets the unique identifier for this graph
    /// </summary>
    string GraphId { get; }

    /// <summary>
    /// Gets the current state of the graph
    /// </summary>
    CUDAGraphState State { get; }

    /// <summary>
    /// Executes the captured graph
    /// </summary>
    /// <param name="stream">CUDA stream for execution</param>
    void Execute(CUDAStream stream);

    /// <summary>
    /// Validates that all captured operations are graph-compatible
    /// </summary>
    /// <returns>Validation result with any errors</returns>
    CUDAGraphValidationResult Validate();
}
```

### 2. ICUDAGraphCapture Interface
Define the interface for graph capture operations.

```csharp
public interface ICUDAGraphCapture : IDisposable
{
    /// <summary>
    /// Gets whether capture is currently active
    /// </summary>
    bool IsCapturing { get; }

    /// <summary>
    /// Begins capturing kernel launches
    /// </summary>
    /// <param name="stream">CUDA stream to capture from</param>
    void BeginCapture(CUDAStream stream);

    /// <summary>
    /// Ends capture and returns the captured graph
    /// </summary>
    /// <returns>The captured CUDA graph</returns>
    ICUDAGraph EndCapture();
}
```

### 3. CUDAGraphState Enum
Define possible states for a CUDA graph.

```csharp
public enum CUDAGraphState
{
    /// <summary>
    /// Graph has been created but not yet captured
    /// </summary>
    Created,

    /// <summary>
    /// Graph is currently being captured
    /// </summary>
    Capturing,

    /// <summary>
    /// Graph has been captured and is ready for execution
    /// </summary>
    Ready,

    /// <summary>
    /// Graph is currently executing
    /// </summary>
    Executing,

    /// <summary>
    /// Graph has been invalidated and cannot be executed
    /// </summary>
    Invalidated,

    /// <summary>
    /// Graph has been disposed
    /// </summary>
    Disposed
}
```

### 4. CUDAGraphValidationResult Class
Define the validation result type.

```csharp
public class CUDAGraphValidationResult
{
    /// <summary>
    /// Gets whether the graph is valid for execution
    /// </summary>
    public bool IsValid { get; init; }

    /// <summary>
    /// Gets list of validation errors (empty if valid)
    /// </summary>
    public IReadOnlyList<string> Errors { get; init; }

    /// <summary>
    /// Gets list of warnings (non-critical issues)
    /// </summary>
    public IReadOnlyList<string> Warnings { get; init; }

    /// <summary>
    /// Gets the number of captured operations
    /// </summary>
    public int OperationCount { get; init; }
}
```

### 5. ICUDAGraphWeightUpdater Interface
Define interface for weight updates within graphs.

```csharp
public interface ICUDAGraphWeightUpdater
{
    /// <summary>
    /// Updates weights in the graph without re-capturing
    /// </summary>
    /// <param name="weightBuffer">Buffer containing updated weights</param>
    /// <param name="offset">Offset in the weight buffer</param>
    /// <param name="size">Size of the weight update</param>
    void UpdateWeights(IntPtr weightBuffer, long offset, long size);

    /// <summary>
    /// Gets the number of weight parameters in the graph
    /// </summary>
    int WeightParameterCount { get; }
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Graphs/ICUDAGraph.cs`
- **File**: `src/CUDA/Graphs/ICUDAGraphCapture.cs`
- **File**: `src/CUDA/Graphs/ICUDAGraphWeightUpdater.cs`
- **File**: `src/CUDA/Graphs/CUDAGraphState.cs`
- **File**: `src/CUDA/Graphs/CUDAGraphValidationResult.cs`

### Dependencies
- Existing CUDA types (CUDAStream, etc.) from the hardware abstraction layer
- System.IDisposable for resource management
- System.Collections.Generic for collections

## Success Criteria
- All interfaces compile without errors
- Interfaces are properly documented with XML documentation
- Proper namespace organization under `RitterFramework.CUDA.Graphs`
- No circular dependencies between interfaces

## Testing Requirements
No unit tests required for this spec (interfaces only).
Tests will be written for implementations in subsequent specs.
