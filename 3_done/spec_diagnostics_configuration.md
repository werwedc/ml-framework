# Spec: Diagnostics Configuration

## Overview
Create a global configuration system to enable/disable enhanced error reporting and control diagnostic behavior.

## Requirements

### Class: DiagnosticsConfiguration
- Location: `src/Core/DiagnosticsConfiguration.cs`

```csharp
public class DiagnosticsConfiguration
{
    // Singleton instance
    public static DiagnosticsConfiguration Instance { get; }

    // Enable/disable enhanced error reporting
    public bool EnhancedErrorReporting { get; set; } = false;

    // Enable/disable suggested fixes
    public bool IncludeSuggestions { get; set; } = true;

    // Enable/disable shape tracking (for performance)
    public bool EnableShapeTracking { get; set; } = false;

    // Maximum depth of context tracking (0 = no context, 1 = immediate parent, etc.)
    public int ContextTrackingDepth { get; set; } = 1;

    // Enable debug mode (more verbose output)
    public bool DebugMode { get; set; } = false;

    // Enable logging of shape information
    public bool LogShapeInformation { get; set; } = false;
}
```

### Global API
- Location: `src/Core/MLFramework.cs`

```csharp
public static class MLFramework
{
    // Enable diagnostics with default settings
    public static void EnableDiagnostics()

    // Enable diagnostics with custom settings
    public static void EnableDiagnostics(DiagnosticsConfiguration config)

    // Disable diagnostics
    public static void DisableDiagnostics()

    // Get current configuration
    public static DiagnosticsConfiguration GetDiagnosticsConfiguration()

    // Check if diagnostics are enabled
    public static bool IsDiagnosticsEnabled()
}
```

### Performance Optimization

**Conditional Compilation:**
```csharp
#if DEBUG
    // Only include shape tracking code in debug builds
    if (DiagnosticsConfiguration.Instance.EnableShapeTracking)
    {
        tensor.SourceOperation = operationName;
        tensor.SourceLayer = layerName;
    }
#endif
```

**Lazy Initialization:**
- Only initialize diagnostic components when first used
- Use lazy loading for registries and formatters

## Integration Points

### With Tensor Operations
```csharp
// In Tensor class methods
public static Tensor MatMul(Tensor a, Tensor b)
{
    if (DiagnosticsConfiguration.Instance.EnhancedErrorReporting)
    {
        var result = ShapeValidator.ValidateMatrixMultiply(a.Shape, b.Shape);
        if (!result.IsValid)
        {
            throw ShapeValidationHelper.CreateException(
                result, OperationType.MatrixMultiply, null, a.Shape, b.Shape);
        }
    }
    // ... rest of implementation
}
```

### With Layer Classes
```csharp
// In Forward methods of layers
public Tensor Forward(Tensor input)
{
    if (DiagnosticsConfiguration.Instance.EnableShapeTracking)
    {
        input.SourceLayer = this.Name;
    }
    // ... rest of implementation
}
```

## Tests
- Create `tests/Core/DiagnosticsConfigurationTests.cs`
- Test singleton pattern
- Test EnableDiagnostics() with default settings
- Test EnableDiagnostics() with custom settings
- Test DisableDiagnostics()
- Test IsDiagnosticsEnabled()
- Test configuration persistence across operations
- Test performance impact with diagnostics disabled

## Success Criteria
- [ ] DiagnosticsConfiguration class with all settings
- [ ] MLFramework API for easy enable/disable
- [ ] Conditional compilation for performance
- [ ] Integration with tensor and layer operations
- [ ] Minimal performance impact when disabled
- [ ] Unit tests pass
