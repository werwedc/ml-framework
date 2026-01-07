# Spec: Calibration Strategies

## Overview
Implement calibration algorithms for determining quantization parameters from calibration data.

## Requirements

### 1. ICalibrator Interface
- Define `ICalibrator` interface with:
  - `CollectStatistics(float[] data)`: Collect statistics from data
  - `GetQuantizationParameters()`: Return QuantizationParameters
  - `Reset()`: Reset calibration state

### 2. MinMaxCalibrator
- Implement `MinMaxCalibrator`:
  - Track min/max values across all data
  - Calculate scale and zero-point from min/max range
  - Handle edge cases (all same values)

### 3. EntropyCalibrator
- Implement `EntropyCalibrator`:
  - Compute histogram of activation values
  - Use KL divergence to find optimal cut-off points
  - Better accuracy for asymmetric distributions

### 4. PercentileCalibrator
- Implement `PercentileCalibrator`:
  - Use configurable percentile (default 99.9%) to exclude outliers
  - More robust to extreme values than min-max
  - Configurable via constructor parameter

### 5. MovingAverageCalibrator
- Implement `MovingAverageCalibrator`:
  - Maintain moving average of min/max values
  - Window size configurable via constructor
  - Useful for dynamic ranges

### 6. Calibration Factory
- Define `CalibratorFactory` with:
  - `Create(CalibrationMethod)`: Return appropriate calibrator instance

## File Structure
```
src/
  MLFramework/
    Quantization/
      Calibration/
        ICalibrator.cs
        MinMaxCalibrator.cs
        EntropyCalibrator.cs
        PercentileCalibrator.cs
        MovingAverageCalibrator.cs
        CalibratorFactory.cs
```

## Implementation Notes
- Use System.Linq for histogram computations
- Entropy calibrator should use KL divergence algorithm from PyTorch
- Handle edge cases (empty data, NaN values)
- Thread-safe for parallel calibration runs

## Dependencies
- Core quantization data structures (spec_quantization_data_structures.md)
- System.Linq, System.Collections.Concurrent
