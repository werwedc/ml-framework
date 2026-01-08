# Spec: DataLoader Configuration

## Overview
Create a robust configuration class for dataloader parameters with validation and sensible defaults.

## Requirements

### 1. DataLoaderConfig Class
Immutable configuration class with validation.

**Properties:**
```csharp
public int NumWorkers { get; }         // Number of parallel workers, default: 4
public int BatchSize { get; }          // Samples per batch, default: 32
public int PrefetchCount { get; }      // Number of batches to prefetch, default: 2
public int QueueSize { get; }          // Max batches in queue, default: 10
public bool Shuffle { get; }           // Randomize data order, default: true
public int Seed { get; }              // Random seed for reproducibility, default: 42
public bool PinMemory { get; }        // Use pinned memory, default: true
```

### 2. Constructor with Validation
```csharp
public DataLoaderConfig(
    int numWorkers = 4,
    int batchSize = 32,
    int prefetchCount = 2,
    int queueSize = 10,
    bool shuffle = true,
    int seed = 42,
    bool pinMemory = true)
```

**Validation Rules:**
- `NumWorkers`: Must be >= 0, defaults to `Environment.ProcessorCount` if 0
- `BatchSize`: Must be > 0
- `PrefetchCount`: Must be >= 0
- `QueueSize`: Must be >= PrefetchCount + 1 (at least one slot for current batch)
- `Seed`: No validation (any int is valid)

**Exceptions:**
- Throw `ArgumentOutOfRangeException` with descriptive message for invalid values
- Message should include parameter name and acceptable range

### 3. Builder Pattern (Optional Enhancement)
Enable fluent configuration for complex setups.

```csharp
public class DataLoaderConfigBuilder
{
    public DataLoaderConfigBuilder WithNumWorkers(int numWorkers)
    public DataLoaderConfigBuilder WithBatchSize(int batchSize)
    public DataLoaderConfigBuilder WithPrefetchCount(int prefetchCount)
    public DataLoaderConfigBuilder WithQueueSize(int queueSize)
    public DataLoaderConfigBuilder WithShuffle(bool shuffle)
    public DataLoaderConfigBuilder WithSeed(int seed)
    public DataLoaderConfigBuilder WithPinMemory(bool pinMemory)
    public DataLoaderConfig Build()
}
```

### 4. Static Factory Methods
Provide convenient presets for common scenarios.

```csharp
public static class DataLoaderConfigPresets
{
    public static DataLoaderConfig ForCPUBound()
    public static DataLoaderConfig ForGPUBound()
    public static DataLoaderConfig ForSmallDataset()
    public static DataLoaderConfig ForLargeDataset()
}
```

**Presets:**
- `ForCPUBound()`: NumWorkers=Environment.ProcessorCount, BatchSize=64, PrefetchCount=3
- `ForGPUBound()`: NumWorkers=2, BatchSize=32, PrefetchCount=2, PinMemory=true
- `ForSmallDataset()`: NumWorkers=1, BatchSize=16, PrefetchCount=1
- `ForLargeDataset()`: NumWorkers=4, BatchSize=128, PrefetchCount=4

### 5. Clone Method
Enable configuration modification without affecting original.

```csharp
public DataLoaderConfig Clone()
public DataLoaderConfig WithNumWorkers(int numWorkers)
public DataLoaderConfig WithBatchSize(int batchSize)
public DataLoaderConfig WithPrefetchCount(int prefetchCount)
public DataLoaderConfig WithQueueSize(int queueSize)
public DataLoaderConfig WithShuffle(bool shuffle)
public DataLoaderConfig WithSeed(int seed)
public DataLoaderConfig WithPinMemory(bool pinMemory)
```

### 6. ToString Override
Human-readable configuration dump for debugging.

```csharp
public override string ToString()
```

**Output Format:**
```
DataLoaderConfig {
  NumWorkers: 4,
  BatchSize: 32,
  PrefetchCount: 2,
  QueueSize: 10,
  Shuffle: True,
  Seed: 42,
  PinMemory: True
}
```

## File Structure
```
src/
  Data/
    DataLoaderConfig.cs         (Main config class)
    DataLoaderConfigBuilder.cs  (Optional builder)
    DataLoaderConfigPresets.cs  (Static factory presets)
```

## Success Criteria
- [ ] All properties defined with correct defaults
- [ ] Constructor validates all parameters
- [ ] ArgumentOutOfRangeException thrown with descriptive messages
- [ ] NumWorkers defaults to processor count when 0
- [ ] QueueSize >= PrefetchCount + 1 enforced
- [ ] Clone method creates independent copy
- [ ] With* methods return new instances (immutable)
- [ ] ToString provides readable output
- [ ] Unit tests cover validation edge cases

## Notes
- Make class immutable (readonly fields, no setters)
- Validation should happen in constructor, not lazily
- Consider XML documentation for public API
- This config will be used throughout the data loading pipeline
