# Feature Idea: Shape Mismatch Error Reporting

## Summary
Enhance the framework's error reporting system to provide descriptive, actionable error messages for tensor shape mismatches. Instead of opaque backend errors, the framework should identify the specific layers, operations, and dimension values involved in shape conflicts, significantly reducing debugging time.

## Problem Statement
Shape mismatch errors are the most common and frustrating bugs in deep learning. When tensors have incompatible shapes, current frameworks often throw generic exceptions with cryptic messages like "RuntimeError: mat1 and mat2 shapes cannot be multiplied" or "Dimension mismatch". These errors force developers to manually trace through the model, add debug prints, or use external tools to identify the problematic operation.

## Proposed Solution
Implement a comprehensive shape mismatch diagnostic system that:

1. **Detailed Shape Information**: For every operation, track and report:
   - Input tensor shapes with explicit dimension sizes
   - Expected output shape based on operation semantics
   - Actual/attempted shape if different
   - Operation type (e.g., Linear, Conv2D, MatrixMultiply)

2. **Contextual Error Messages**: Generate human-readable messages that include:
   - The specific layer/module name where the error occurred
   - The operation being performed
   - Exact shape requirements (e.g., "Expected input shape [batch_size, 784], got [32, 100]")
   - Suggested fixes based on common patterns (e.g., "Consider adding a reshape layer")

3. **Operation-Specific Diagnostics**: Provide specialized messages for different operation types:
   - **Linear layers**: Show weight matrix shape vs input shape
   - **Convolution**: Show kernel size, stride, padding calculations
   - **Concatenation/Stacking**: Show all input shapes and the mismatch
   - **Reductions**: Show which dimension and expected size

4. **Shape Inference Visualization**: Optionally provide a visual representation of the tensor flow with shapes at each step, similar to a computation graph but annotated with dimension information.

5. **Common Pattern Detection**: Detect and suggest fixes for common issues:
   - Missing or incorrect batch dimensions
   - Channel order mismatches (NCHW vs NHWC)
   - Broadcasting failures
   - Transpose requirements

## Example Error Messages

### Before (Generic):
```
System.InvalidOperationException: Tensor dimension mismatch in matrix multiplication
```

### After (Enhanced):
```
MLFramework.ShapeMismatchException: Matrix multiplication failed in layer 'encoder.fc2'

Input shape:    [32, 256]
Weight shape:   [128, 10]
Expected:       [batch_size, input_features] × [input_features, output_features]
                → Requires input_features to match

Problem: Dimension 1 of input (256) does not match dimension 0 of weight (128)

Context:
- Layer: encoder.fc2 (Linear)
- Batch size: 32
- Previous layer output: encoder.fc1 with shape [32, 256]

Suggested fixes:
1. Check encoder.fc1 output features (currently 256) matches fc2 input features (expected 128)
2. Consider adjusting fc1 to output 128 features, or fc2 to accept 256 inputs
3. Verify model configuration matches expected architecture
```

## Technical Implementation

### Core Components

1. **ShapeTrackingModule**: A wrapper or base class that tracks tensor shapes through the computational graph
2. **OperationMetadataRegistry**: Central registry of shape requirements for all operations
3. **ShapeMismatchException**: Custom exception type with rich diagnostic information
4. **ShapeInferenceEngine**: Optional component that can predict shapes without executing operations

### API Design

```csharp
// Enable enhanced error reporting
MLFramework.EnableDiagnostics();

try
{
    var output = model.Forward(input);
}
catch (ShapeMismatchException ex)
{
    // Access rich diagnostic information
    Console.WriteLine(ex.LayerName);           // "encoder.fc2"
    Console.WriteLine(ex.OperationType);       // OperationType.MatrixMultiply
    Console.WriteLine(ex.InputShapes);         // [32, 256]
    Console.WriteLine(ex.ExpectedShapes);      // [*, 128]
    Console.WriteLine(ex.ProblemDescription);  // "Dimension 1 mismatch..."
    Console.WriteLine(ex.SuggestedFixes);      // List of suggestions

    // Or get a formatted report
    Console.WriteLine(ex.GetDiagnosticReport());

    // Optional: Visualize the shape flow
    ex.VisualizeShapeFlow(outputPath: "shape_error.html");
}

// Programmatic shape checking
if (!MLFramework.CheckShapes(operation, inputTensor, weightTensor))
{
    var diagnostics = MLFramework.GetShapeDiagnostics(operation, inputTensor, weightTensor);
    // Log warning or handle proactively
}
```

## Benefits

1. **Reduced Debugging Time**: Developers can identify shape issues in seconds rather than hours
2. **Better DX (Developer Experience)**: Makes the framework more approachable for beginners
3. **Fewer Support Requests**: Clear error messages reduce the need for external help
4. **Educational Value**: Helps users understand tensor operations and model architecture
5. **Professionalism**: Positions the framework as a mature, production-ready tool

## Implementation Phases

### Phase 1: Basic Shape Tracking (MVP)
- Add shape metadata to tensor objects
- Capture operation context at error sites
- Generate basic descriptive error messages

### Phase 2: Rich Diagnostics
- Implement operation-specific shape requirements registry
- Add suggested fixes based on pattern matching
- Create formatted error reports

### Phase 3: Advanced Features
- Optional shape inference without execution
- Visual shape flow diagrams
- Integration with the visualizer/profiler
- Interactive debugging tools

### Phase 4: Ecosystem Integration
- IDE plugin support (highlight problematic lines)
- Shape-aware autocomplete in model builders
- Automated shape checking in CI/CD pipelines

## Related Features

- **Visualizer/Profiler Integration**: Can visualize shape flow alongside computation graphs (0_ideas/visualizer_profiler_integration.md)
- **Functional Transformations**: Shape inference aids in vmap and other transforms (0_ideas/functional_transformations.md)
- **Hardware Abstraction Layer**: HAL can provide shape-related performance warnings

## Considerations

1. **Performance Impact**: Shape tracking should have minimal overhead when not in use (compile-time or debug-only)
2. **Opt-out Option**: Advanced users may prefer concise errors; provide a flag to disable verbose diagnostics
3. **Extensibility**: Allow custom operations to define their own shape rules and error messages
4. **Localization**: Consider supporting error messages in multiple languages

## Success Metrics

- Reduced average time to resolve shape-related bugs (target: 70% reduction)
- Improved user satisfaction scores in developer surveys
- Decrease in shape-related questions in forums/issue trackers
- Adoption in tutorials and educational materials due to clarity

## Open Questions

1. Should shape tracking be enabled by default in debug mode, or opt-in only?
2. How deep should the context追溯 go? (immediate parent only, or full call stack?)
3. Should we provide an interactive REPL-style debugger for shape issues?
4. Integration with existing .NET diagnostics tools (e.g., Visual Studio debugger)?
