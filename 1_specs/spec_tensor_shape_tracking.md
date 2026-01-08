# Spec: Tensor Shape Tracking

## Overview
Enhance Tensor objects to track and expose shape metadata for diagnostic purposes.

## Requirements

### Modifications to Tensor Class
- Location: `src/Core/Tensor.cs`

### New Properties
```csharp
// Track the shape at tensor creation time
public long[] Shape { get; private set; }

// Track the operation that created this tensor (optional)
public string SourceOperation { get; set; }

// Track the layer/module that created this tensor (optional)
public string SourceLayer { get; set; }
```

### New Methods
```csharp
// Get shape as formatted string
public string GetShapeString()

// Get dimension count
public int GetRank()

// Get size of specific dimension
public long GetDimension(int index)

// Check if shape matches another tensor
public bool HasSameShape(Tensor other)
```

### Internal Tracking
- Update Shape property when tensor is created
- Allow setting SourceOperation and SourceLayer for tracking purposes
- Ensure shape is immutable after creation

## Tests
- Create/update `tests/Core/TensorTests.cs`
- Test shape tracking initialization
- Test GetShapeString() formatting
- Test GetRank() for different tensor dimensions
- Test GetDimension() with valid and invalid indices
- Test HasSameShape() comparison

## Success Criteria
- [ ] Tensor class tracks shape metadata
- [ ] Helper methods work correctly
- [ ] Unit tests pass
- [ ] Minimal performance impact (benchmark if needed)
