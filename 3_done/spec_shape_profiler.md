# Spec: Shape Profiler

## Overview
Implement shape profiling to track actual shape distributions and inform optimization decisions.

## Requirements

### Class: ShapeSample
- Properties:
  - `Shape`: int[]
  - `Timestamp`: DateTime
  - `TensorName`: string
  - `OperationName`: string

### Class: ShapeHistogram
- Properties:
  - `BinCounts`: Dictionary<string, int> - String representation of shape -> count
  - `TotalSamples`: int
  - `UniqueShapes`: int
  - `MostCommonShape`: int[]?
  - `MostCommonCount`: int

- Methods:
  - `AddSample(int[] shape)`: void
  - `GetFrequency(int[] shape)`: double
  - `GetTopShapes(int count)`: List<(int[] shape, int count)>
  - `GetProbability(int[] shape)`: double
  - `ToReport()`: string

### Interface: IShapeProfiler
- Methods:
  - `RecordShape(string tensorName, string opName, int[] shape)`: void
  - `GetHistogram(string tensorName)`: ShapeHistogram?
  - `GetCommonShapes(string tensorName, int count)`: List<int[]>
  - `GetShapeStatistics(string tensorName)`: ShapeStatistics?
  - `Clear(string tensorName)`: void
  - `ClearAll()`: void

### Class: GlobalShapeProfiler : IShapeProfiler
- Properties:
  - `TensorHistograms`: Dictionary<string, ShapeHistogram>
  - `MaxSamplesPerTensor`: int (default 10000)

- Methods:
  - Thread-safe shape recording
  - Automatic sampling if samples exceed limit (reservoir sampling)
  - `GetReport()`: string - Full profiling report

### Class: ShapeStatistics
- Properties:
  - `MeanShape`: double[]
  - `StdDevShape`: double[]
  - `MinShape`: int[]
  - `MaxShape`: int[]
  - `Percentiles`: Dictionary<int, double[]> - e.g., 25th, 50th, 75th percentiles

- Methods:
  - `CalculateFromHistogram(ShapeHistogram histogram)`: void

### Class: ShapeProfileOptimizer
- Methods:
  - `RecommendSpecializedShapes(string tensorName, int threshold)`: List<int[]> - Shapes worth specializing
  - `ShouldRecompile(int[] newShape, ShapeHistogram histogram, double threshold)`: bool
  - `GetOptimalPadding(ShapeHistogram histogram)`: int[] - Suggested bounds

### Unit Tests
- Test histogram sampling
- Test statistics calculation
- Test profiler recording
- Test recommendation logic
- Test recompilation decisions
- Test reservoir sampling when limit exceeded

## Implementation Notes
- Use lock-free data structures where possible for performance
- Sampling should have minimal overhead during execution
- Persist profiles to disk for cross-session learning
- Provide hooks for custom analysis
- Support filtering by time range or operation

## Dependencies
- spec_symbolic_shape.md

## Success Criteria
- Tracks shape distributions accurately
- Statistics are calculated correctly
- Recommendations guide compilation decisions
- Minimal overhead during execution
- Profiles persist across sessions
