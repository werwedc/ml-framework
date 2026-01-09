# Spec: Validation and Diagnostics

## Overview
Implement validation utilities and diagnostic tools to detect non-deterministic operations, warn about non-determinizable operations, and verify reproducibility across runs. This spec adds awareness and transparency to the deterministic reproducibility system.

## Technical Requirements

### Validation Classes
```csharp
namespace MLFramework.Utilities;

/// <summary>
/// Validation result for deterministic behavior
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// Whether validation passed
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// List of validation messages
    /// </summary>
    public List<string> Messages { get; set; }

    /// <summary>
    /// List of warnings
    /// </summary>
    public List<string> Warnings { get; set; }

    /// <summary>
    /// List of errors
    /// </summary>
    public List<string> Errors { get; set; }

    public ValidationResult()
    {
        Messages = new List<string>();
        Warnings = new List<string>();
        Errors = new List<string>();
    }

    public bool HasWarnings => Warnings.Count > 0;
    public bool HasErrors => Errors.Count > 0;
}

/// <summary>
/// Diagnostic information about current deterministic state
/// </summary>
public class DiagnosticInfo
{
    /// <summary>
    /// Current deterministic mode flags
    /// </summary>
    public DeterministicModeFlags DeterministicMode { get; set; }

    /// <summary>
    /// Current global seed
    /// </summary>
    public int CurrentSeed { get; set; }

    /// <summary>
    /// Number of devices seeded
    /// </summary>
    public int DeviceCount { get; set; }

    /// <summary>
    /// Number of workers seeded
    /// </summary>
    public int WorkerCount { get; set; }

    /// <summary>
    /// Performance impact estimate
    /// </summary>
    public string PerformanceImpact { get; set; }

    /// <summary>
    /// Known non-deterministic operations in use
    /// </summary>
    public List<string> NonDeterministicOperations { get; set; }

    public DiagnosticInfo()
    {
        NonDeterministicOperations = new List<string>();
    }
}
```

### Extended SeedManager Methods
```csharp
public class SeedManager : IDisposable
{
    // ... existing methods ...

    private readonly List<string> _nonDeterministicOperations = new();

    /// <summary>
    /// Registers a known non-deterministic operation
    /// </summary>
    /// <param name="operationName">Name of the operation</param>
    public void RegisterNonDeterministicOperation(string operationName);

    /// <summary>
    /// Gets a list of registered non-deterministic operations
    /// </summary>
    /// <returns>List of operation names</returns>
    public IReadOnlyList<string> GetNonDeterministicOperations();

    /// <summary>
    /// Validates current deterministic configuration
    /// </summary>
    /// <returns>ValidationResult with messages, warnings, and errors</returns>
    public ValidationResult ValidateConfiguration();

    /// <summary>
    /// Gets diagnostic information about current state
    /// </summary>
    /// <returns>DiagnosticInfo object</returns>
    public DiagnosticInfo GetDiagnosticInfo();

    /// <summary>
    /// Prints diagnostic information to console
    /// </summary>
    public void PrintDiagnostics();

    /// <summary>
    /// Checks if an operation can be made deterministic
    /// </summary>
    /// <param name="operationName">Name of the operation</param>
    /// <returns>True if operation can be deterministic</returns>
    public bool CanBeDeterministic(string operationName);

    /// <summary>
    /// Warns about performance impact of current deterministic settings
    /// </summary>
    /// <returns>Warning message or null if no impact</returns>
    public string? CheckPerformanceImpact();
}
```

### Known Non-Deterministic Operations
```csharp
public static class KnownNonDeterministicOperations
{
    /// <summary>
    /// Operations that cannot be made deterministic
    /// </summary>
    public static readonly IReadOnlySet<string> NonDeterminizable = new HashSet<string>
    {
        "atomic_add",
        "atomic_sub",
        "scatter_add",
        "scatter_sub",
        "parallel_sort",  // Some sorting algorithms are non-deterministic
        "hashmap_lookup"  // Hash-based operations
    };

    /// <summary>
    /// Operations that can be made deterministic with performance impact
    /// </summary>
    public static readonly IReadOnlySet<string> Determinizable = new HashSet<string>
    {
        "convolution",
        "matmul",
        "dropout",
        "batch_norm",
        "shuffle",
        "random_sample"
    };
}
```

### Implementation Details

#### RegisterNonDeterministicOperation Method
- Add operation name to _nonDeterministicOperations list
- Avoid duplicates
- Log registration for debugging

#### GetNonDeterministicOperations Method
- Return read-only copy of _nonDeterministicOperations
- Provide visibility into what non-deterministic operations are in use

#### ValidateConfiguration Method
- Create new ValidationResult
- Check if deterministic mode is enabled
- Validate all devices have been seeded if multi-GPU is used
- Validate all workers have been seeded if multi-worker is used
- Check for non-deterministic operations in deterministic mode
- Return validation result with appropriate messages, warnings, and errors

#### GetDiagnosticInfo Method
- Create new DiagnosticInfo
- Populate with current deterministic mode
- Populate with current seed
- Populate with device and worker counts
- Populate with performance impact from GetPerformanceImpact()
- Populate with list of non-deterministic operations
- Return DiagnosticInfo

#### PrintDiagnostics Method
- Call GetDiagnosticInfo()
- Format and print to console with clear sections
- Use colors or formatting for warnings and errors
- Include timestamp for logging purposes

#### CanBeDeterministic Method
- Check if operation is in NonDeterminizable set
- Return false if operation cannot be deterministic
- Return true otherwise (operation can be deterministic)

#### CheckPerformanceImpact Method
- Check current deterministic mode flags
- Return warning message if performance-critical flags are set
- Return null if deterministic mode is disabled
- Example warnings:
  - "cuDNN deterministic mode may reduce performance by 20-30%"
  - "cuBLAS deterministic mode may reduce performance by 15-25%"

### Design Decisions

1. **Operation Registry**: Track non-deterministic operations at runtime
2. **Validation Layer**: Separate validation from core functionality
3. **Diagnostic Information**: Comprehensive state reporting
4. **Known Operations**: Maintain lists of determinizable and non-determinizable operations
5. **Performance Transparency**: Warn users about performance trade-offs

### Validation Rules

#### Configuration Validation
- If deterministic mode is enabled:
  - All devices must be seeded
  - All workers must be seeded
  - No non-determinizable operations should be used
  - Warn about determinizable operations in use

#### Reproducibility Validation
- Same seed should produce same results
- State capture/restore should be idempotent
- Multi-device seeding should be consistent

### Warning Scenarios

#### Performance Warnings
- Deterministic mode enabled in production
- cuDNN deterministic mode enabled
- cuBLAS deterministic mode enabled
- Multiple devices seeded (potential performance overhead)

#### Non-Deterministic Operation Warnings
- Using non-determinizable operations in deterministic mode
- Using determinizable operations without proper deterministic settings

### Directory Structure
```
src/
  MLFramework.Utilities/
    SeedManager.cs (extended)
    ValidationResult.cs
    DiagnosticInfo.cs
    KnownNonDeterministicOperations.cs
```

## Success Criteria
1. ✅ RegisterNonDeterministicOperation tracks operations correctly
2. ✅ GetNonDeterministicOperations returns registered operations
3. ✅ ValidateConfiguration detects configuration issues
4. ✅ GetDiagnosticInfo returns comprehensive state information
5. ✅ PrintDiagnostics outputs readable diagnostic information
6. ✅ CanBeDeterministic correctly identifies deterministic capability
7. ✅ CheckPerformanceImpact provides appropriate warnings

## Dependencies
- spec_seed_manager_core.md - Base SeedManager class
- spec_deterministic_mode.md - DeterministicModeFlags
- spec_multidevice_seeding.md - Multi-device seeding methods

## Notes
- Operation registry is manual; consider automatic detection in future
- Validation warnings are informational, not blocking
- Consider integrating with logging framework for diagnostic output

## Related Specs
- spec_seed_manager_core.md - Base class that this extends
- spec_deterministic_mode.md - Provides deterministic mode information
- spec_multidevice_seeding.md - Provides device/worker information for validation
