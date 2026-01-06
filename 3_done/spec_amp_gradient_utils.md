# Spec: AMP Gradient Utilities

## Overview
Implement utility functions for gradient unscaling and overflow checking in Automatic Mixed Precision training.

## Class Specification

### 1. GradientUtils Class

**File:** `src/MLFramework/Amp/GradientUtils.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Utility functions for gradient manipulation in AMP
    /// </summary>
    public static class GradientUtils
    {
        /// <summary>
        /// Unscales a gradient tensor by dividing by the scale factor
        /// </summary>
        /// <param name="gradient">The gradient tensor to unscale</param>
        /// <param name="scale">The scale factor to divide by</param>
        /// <returns>Unscaled gradient tensor</returns>
        public static Tensor Unscale(Tensor gradient, float scale);

        /// <summary>
        /// Unscales multiple gradient tensors
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <param name="scale">The scale factor to divide by</param>
        /// <returns>Dictionary of unscaled gradient tensors</returns>
        public static Dictionary<string, Tensor> Unscale(
            Dictionary<string, Tensor> gradients,
            float scale);

        /// <summary>
        /// In-place unscale of a gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to unscale in-place</param>
        /// <param name="scale">The scale factor to divide by</param>
        public static void UnscaleInPlace(Tensor gradient, float scale);

        /// <summary>
        /// Checks for overflow (Inf/NaN) in a gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to check</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        public static bool CheckOverflow(Tensor gradient);

        /// <summary>
        /// Checks for overflow (Inf/NaN) in multiple gradient tensors
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>True if any overflow detected, false otherwise</returns>
        public static bool CheckOverflow(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Checks for overflow (Inf/NaN) in an array of tensors
        /// </summary>
        /// <param name="tensors">Array of tensors to check</param>
        /// <returns>True if any overflow detected, false otherwise</returns>
        public static bool CheckOverflow(Tensor[] tensors);

        /// <summary>
        /// Checks if a tensor contains Inf values
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <returns>True if any Inf values, false otherwise</returns>
        public static bool IsInf(Tensor tensor);

        /// <summary>
        /// Checks if a tensor contains NaN values
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <returns>True if any NaN values, false otherwise</returns>
        public static bool IsNaN(Tensor tensor);

        /// <summary>
        /// Checks if a tensor contains any Inf or NaN values
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <returns>True if any Inf or NaN values, false otherwise</returns>
        public static bool IsInfOrNaN(Tensor tensor);

        /// <summary>
        /// Finds tensors with overflow in a dictionary
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>List of parameter names with overflow</returns>
        public static List<string> FindOverflowGradients(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Gets statistics about gradient overflow
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>Overflow statistics</returns>
        public static OverflowStats GetOverflowStats(Dictionary<string, Tensor> gradients);
    }
}
```

### 2. OverflowStats Class

**File:** `src/MLFramework/Amp/OverflowStats.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Statistics about gradient overflow
    /// </summary>
    public class OverflowStats
    {
        /// <summary>
        /// Gets the total number of gradients checked
        /// </summary>
        public int TotalGradients { get; }

        /// <summary>
        /// Gets the number of gradients with overflow
        /// </summary>
        public int OverflowCount { get; }

        /// <summary>
        /// Gets the list of parameter names with overflow
        /// </summary>
        public IReadOnlyList<string> OverflowParameters { get; }

        /// <summary>
        /// Gets the overflow rate (overflow count / total count)
        /// </summary>
        public float OverflowRate { get; }

        /// <summary>
        /// Gets whether any overflow was detected
        /// </summary>
        public bool HasOverflow => OverflowCount > 0;

        /// <summary>
        /// Creates a new OverflowStats
        /// </summary>
        public OverflowStats(
            int totalGradients,
            int overflowCount,
            IReadOnlyList<string> overflowParameters);

        /// <summary>
        /// Returns a string representation of the statistics
        /// </summary>
        public override string ToString();
    }
}
```

### 3. GradientClipper Class

**File:** `src/MLFramework/Amp/GradientClipper.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Gradient clipping utilities for AMP
    /// </summary>
    public static class GradientClipper
    {
        /// <summary>
        /// Clips gradients by value (clamp between -clipValue and +clipValue)
        /// </summary>
        /// <param name="gradient">The gradient tensor to clip</param>
        /// <param name="clipValue">The maximum absolute value</param>
        /// <returns>Clipped gradient tensor</returns>
        public static Tensor ClipByValue(Tensor gradient, float clipValue);

        /// <summary>
        /// Clips gradients by norm
        /// </summary>
        /// <param name="gradient">The gradient tensor to clip</param>
        /// <param name="maxNorm">The maximum L2 norm</param>
        /// <param name="normType">The type of norm (default: L2)</param>
        /// <returns>Clipped gradient tensor</returns>
        public static Tensor ClipByNorm(Tensor gradient, float maxNorm, float normType = 2.0f);

        /// <summary>
        /// Clips multiple gradients by norm
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <param name="maxNorm">The maximum L2 norm</param>
        /// <param name="normType">The type of norm (default: L2)</param>
        /// <returns>Dictionary of clipped gradient tensors</returns>
        public static Dictionary<string, Tensor> ClipByNorm(
            Dictionary<string, Tensor> gradients,
            float maxNorm,
            float normType = 2.0f);

        /// <summary>
        /// Computes the gradient norm
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <param name="normType">The type of norm (default: L2)</param>
        /// <returns>The gradient norm</returns>
        public static float ComputeNorm(
            Dictionary<string, Tensor> gradients,
            float normType = 2.0f);
    }
}
```

## Implementation Details

### Unscale Implementation

```csharp
public static Tensor Unscale(Tensor gradient, float scale)
{
    if (scale == 1.0f)
    {
        return gradient;
    }

    // Create inverse scale tensor
    var inverseScale = Tensor.Scalar(1.0f / scale, gradient.Dtype);

    // Divide gradient by scale
    return gradient * inverseScale;
}

public static Dictionary<string, Tensor> Unscale(
    Dictionary<string, Tensor> gradients,
    float scale)
{
    if (scale == 1.0f)
    {
        return gradients;
    }

    var result = new Dictionary<string, Tensor>();
    foreach (var (name, grad) in gradients)
    {
        result[name] = Unscale(grad, scale);
    }
    return result;
}
```

### Overflow Detection Implementation

```csharp
public static bool CheckOverflow(Tensor gradient)
{
    return IsInf(gradient) || IsNaN(gradient);
}

public static bool CheckOverflow(Dictionary<string, Tensor> gradients)
{
    // Early exit on first overflow
    foreach (var grad in gradients.Values)
    {
        if (CheckOverflow(grad))
        {
            return true;
        }
    }
    return false;
}

public static bool IsInf(Tensor tensor)
{
    // Use tensor operations to check for Inf
    var isInfTensor = tensor.IsInfinity();
    return isInfTensor.Any();
}

public static bool IsNaN(Tensor tensor)
{
    // Use tensor operations to check for NaN
    var isNaNTensor = tensor.IsNaN();
    return isNaNTensor.Any();
}
```

### Gradient Clipping Implementation

```csharp
public static Tensor ClipByNorm(Tensor gradient, float maxNorm, float normType = 2.0f)
{
    var norm = gradient.Norm(normType);
    var scale = maxNorm / (norm + 1e-6f); // Add epsilon for numerical stability
    scale = Math.Min(scale, 1.0f); // Don't increase gradients
    return gradient * scale;
}

public static float ComputeNorm(
    Dictionary<string, Tensor> gradients,
    float normType = 2.0f)
{
    float totalNorm = 0.0f;
    foreach (var grad in gradients.Values)
    {
        float norm = grad.Norm(normType).ToScalar();
        totalNorm += MathF.Pow(norm, normType);
    }
    return MathF.Pow(totalNorm, 1.0f / normType);
}
```

### Overflow Stats Implementation

```csharp
public static OverflowStats GetOverflowStats(Dictionary<string, Tensor> gradients)
{
    int totalCount = gradients.Count;
    int overflowCount = 0;
    var overflowParams = new List<string>();

    foreach (var (name, grad) in gradients)
    {
        if (CheckOverflow(grad))
        {
            overflowCount++;
            overflowParams.Add(name);
        }
    }

    return new OverflowStats(totalCount, overflowCount, overflowParams);
}
```

## Usage Examples

### Basic Unscaling
```csharp
var grads = model.GetGradients();
var unscaledGrads = GradientUtils.Unscale(grads, scale);
optimizer.Step(unscaledGrads);
```

### Overflow Checking
```csharp
var grads = model.GetGradients();
bool hasOverflow = GradientUtils.CheckOverflow(grads);

if (hasOverflow)
{
    var overflowParams = GradientUtils.FindOverflowGradients(grads);
    Console.WriteLine($"Overflow in: {string.Join(", ", overflowParams)}");
}
```

### Gradient Clipping
```csharp
var grads = model.GetGradients();
var clippedGrads = GradientClipper.ClipByNorm(grads, maxNorm: 1.0f);
optimizer.Step(clippedGrads);
```

### Combined: Unscale + Clip + Check
```csharp
var grads = model.GetGradients();
var unscaledGrads = GradientUtils.Unscale(grads, scale);

if (!GradientUtils.CheckOverflow(unscaledGrads))
{
    var clippedGrads = GradientClipper.ClipByNorm(unscaledGrads, maxNorm: 1.0f);
    optimizer.Step(clippedGrads);
}
```

## Dependencies
- MLFramework.Core (Tensor)
- System.Collections.Generic (Dictionary, List, IReadOnlyList)
- System (Math, MathF)

## Testing Requirements
- Test Unscale with various scale factors
- Test UnscaleInPlace modifies tensor correctly
- Test CheckOverflow detects Inf values
- Test CheckOverflow detects NaN values
- Test CheckOverflow with multiple tensors
- Test FindOverflowGradients returns correct parameters
- Test GetOverflowStats calculates correct statistics
- Test GradientClipper.ClipByValue
- Test GradientClipper.ClipByNorm
- Test GradientClipper.ComputeNorm
- Test edge cases (scale = 0, scale = Inf, empty gradients)

## Success Criteria
- [ ] Unscale divides by scale factor correctly
- [ ] UnscaleInPlace modifies tensor in-place
- [ ] CheckOverflow detects Inf and NaN
- [ ] FindOverflowGradients identifies correct parameters
- [ ] GetOverflowStats returns accurate statistics
- [ ] Gradient clipping works as expected
- [ ] ComputeNorm returns correct values
- [ ] Performance overhead < 5% of gradient computation time
- [ ] All unit tests pass
- [ ] Documentation includes usage examples
