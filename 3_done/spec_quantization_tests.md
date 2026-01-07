# Spec: Quantization Unit Tests

## Overview
Implement unit tests for core quantization functionality.

## Requirements

### 1. QuantizationParameters Tests
- Test `QuantizationParameters` struct:
  - Initialization with valid values
  - Scale and zero-point calculations
  - Edge cases (zero scale, extreme values)

### 2. QuantizationMode Tests
- Test quantization mode enum:
  - All mode values are valid
  - Mode string representation
  - Mode parsing from string

### 3. CalibrationMethod Tests
- Test calibration method enum:
  - All method values are valid
  - Method string representation
  - Method parsing from string

### 4. QuantizationConfig Tests
- Test `QuantizationConfig` class:
  - Valid configuration creation
  - Validation of invalid configurations
  - Default configuration values
  - Clone/copy operations

### 5. Quantize/Dequantize Tests
- Test quantization operations:
  - Round-trip: quantize + dequantize ≈ original
  - Clamping to Int8 range [-128, 127]
  - Rounding behavior (half away from zero)
  - Symmetric vs asymmetric quantization
  - Edge cases (NaN, Inf, all same values)

### 6. Tensor Quantization Tests
- Test tensor quantization:
  - Per-tensor quantization
  - Per-channel quantization
  - Correct scale/zero-point computation
  - Correct clamping behavior

### 7. MinMaxCalibrator Tests
- Test `MinMaxCalibrator`:
  - Correct min/max tracking
  - Correct scale/zero-point calculation
  - Empty data handling
  - Single value handling

### 8. EntropyCalibrator Tests
- Test `EntropyCalibrator`:
  - Histogram computation accuracy
  - KL divergence calculation
  - Optimal cut-off point selection
  - Comparison with min-max results

### 9. PercentileCalibrator Tests
- Test `PercentileCalibrator`:
  - Correct percentile computation
  - Outlier exclusion
  - Configurable percentile values
  - Edge cases (all outliers)

### 10. MovingAverageCalibrator Tests
- Test `MovingAverageCalibrator`:
  - Correct moving average computation
  - Window size behavior
  - Reset functionality
  - Multiple batches handling

## File Structure
```
tests/
  MLFramework.Tests/
    Quantization/
      DataStructures/
        QuantizationParametersTests.cs
        QuantizationModeTests.cs
        CalibrationMethodTests.cs
        QuantizationConfigTests.cs
      Operations/
        QuantizationOperationsTests.cs
        DequantizationOperationsTests.cs
        PerChannelOperationsTests.cs
      Calibration/
        MinMaxCalibratorTests.cs
        EntropyCalibratorTests.cs
        PercentileCalibratorTests.cs
        MovingAverageCalibratorTests.cs
```

## Implementation Notes
- Use xUnit test framework
- Use theory tests for parameterized testing
- Include edge cases and boundary conditions
- Use float tolerance assertions (approximate equality)

## Dependencies
- Core quantization components
- xUnit test framework
- FluentAssertions for better assertions
