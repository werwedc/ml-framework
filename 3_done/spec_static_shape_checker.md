# Spec: Static Shape Checker

## Overview
Implement early detection of shape incompatibilities before execution to catch errors at compile time.

## Requirements

### Class: ShapeMismatchException : Exception
- Properties:
  - `OperationName`: string
  - `ExpectedShapes`: List<SymbolicShape>
  - `ActualShapes`: List<SymbolicShape>
  - `Details`: string

### Class: StaticShapeChecker
- Methods:
  - `CheckOperation(Operation op, List<SymbolicShape> inputs)`: void - Throws on invalid
  - `CheckSequence(List<Operation> ops, Dictionary<string, SymbolicShape> tensorShapes)`: void
  - `CheckBroadcastCompatibility(SymbolicShape a, SymbolicShape b)`: void
  - `CheckReshapeValid(SymbolicShape from, SymbolicShape to)`: void
  - `CheckMatMulCompatibility(SymbolicShape a, SymbolicShape b)`: void

- Internal validation methods:
  - `CheckRank(Operation op, List<SymbolicShape> inputs, int expectedRank)`: void
  - `CheckDim(Operation op, SymbolicShape shape, int dimIndex, int expectedValue)`: void
  - `CheckDimRange(Operation op, SymbolicShape shape, int dimIndex, int min, int max)`: void

### Class: ShapeErrorReporter
- Methods:
  - `FormatError(ShapeMismatchException ex)`: string - User-friendly message
  - `SuggestFix(ShapeMismatchException ex)`: string - Hint on how to resolve
  - `VisualizeShapes(List<SymbolicShape> shapes)`: string - ASCII representation

### Unit Tests
- Test each check method with valid and invalid inputs
- Test exception properties are populated correctly
- Test error reporter formatting
- Test suggestion generation for common errors
- Test sequence checking across multiple operations

## Implementation Notes
- Perform aggressive checking - prefer false positives over runtime errors
- Use symbolic bounds when concrete values unknown (e.g., if dim >= 1 required, check min bound)
- Accumulate all errors in sequence check before throwing
- Provide context in error messages (which operation, which tensors)

## Dependencies
- spec_shape_inference_engine.md

## Success Criteria
- Catches common shape errors before execution
- Error messages are actionable and informative
- Checks work for both concrete and symbolic shapes
- Performance overhead is minimal for eager checking
