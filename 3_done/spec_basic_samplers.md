# Spec: Basic Sampler Implementations

## Overview
Implement core sampler classes that determine how samples are selected from a dataset.

## Requirements

### Interface

#### ISampler
```csharp
public interface ISampler
{
    IEnumerable<int> Iterate();
    int Length { get; }
}
```

### Implementations

#### SequentialSampler
- Returns indices in sequential order: 0, 1, 2, ..., N-1
- Deterministic and repeatable
- No randomness involved
- Useful for validation and testing

**Key Methods:**
```csharp
public class SequentialSampler : ISampler
{
    private readonly int _size;

    public SequentialSampler(int size)
    {
        if (size < 0)
            throw new ArgumentOutOfRangeException(nameof(size));

        _size = size;
        Length = size;
    }

    public int Length { get; }

    public IEnumerable<int> Iterate()
    {
        for (int i = 0; i < _size; i++)
            yield return i;
    }
}
```

#### RandomSampler
- Returns indices in random order without replacement
- Uses `System.Random` for reproducibility
- Configurable seed for deterministic behavior
- Guarantees each index appears exactly once per epoch

**Key Methods:**
```csharp
public class RandomSampler : ISampler
{
    private readonly int _size;
    private readonly Random _random;

    public RandomSampler(int size, int? seed = null)
    {
        if (size < 0)
            throw new ArgumentOutOfRangeException(nameof(size));

        _size = size;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        Length = size;
    }

    public int Length { get; }

    public IEnumerable<int> Iterate()
    {
        var indices = Enumerable.Range(0, _size).ToList();
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
        return indices;
    }
}
```

### Error Handling
- `ArgumentOutOfRangeException` for negative dataset sizes
- Validation in constructors

## Acceptance Criteria
1. SequentialSampler returns indices 0 to N-1 in order
2. RandomSampler returns all indices exactly once, shuffled
3. RandomSampler with same seed produces same order
4. Both samplers correctly report Length property
5. Empty dataset returns empty iterator
6. Unit tests verify reproducibility of RandomSampler

## Files to Create
- `src/Data/ISampler.cs`
- `src/Data/SequentialSampler.cs`
- `src/Data/RandomSampler.cs`

## Tests
- `tests/Data/SequentialSamplerTests.cs`
- `tests/Data/RandomSamplerTests.cs`

## Notes
- RandomSampler uses Fisher-Yates shuffle for efficiency (O(n))
- Implement as lazy evaluation (yield return) to avoid pre-allocation
- Thread safety: Create new sampler per thread/process if needed
