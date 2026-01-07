# Spec: Core Quantization Data Structures

## Overview
Define fundamental data structures for quantization operations including quantization parameters, modes, and configuration.

## Requirements

### 1. Quantization Parameters
- Define `QuantizationParameters` struct with:
  - `Scale` (float32): Scale factor for quantization
  - `ZeroPoint` (int32): Zero-point offset
  - `Min` (float32): Original min value (for calibration)
  - `Max` (float32): Original max value (for calibration)

### 2. Quantization Mode Enum
- Define `QuantizationMode` enum:
  - `PerTensorSymmetric`: Single scale, zero-point = 0
  - `PerTensorAsymmetric`: Single scale, single zero-point
  - `PerChannelSymmetric`: Per-channel scale, zero-point = 0
  - `PerChannelAsymmetric`: Per-channel scale, per-channel zero-point

### 3. Calibration Method Enum
- Define `CalibrationMethod` enum:
  - `MinMax`: Min-max calibration
  - `Entropy`: Entropy-based calibration
  - `Percentile`: Percentile-based calibration
  - `MovingAverage`: Moving average calibration

### 4. Quantization Type Enum
- Define `QuantizationType` enum:
  - `Int8`: 8-bit signed integer
  - `UInt8`: 8-bit unsigned integer

### 5. Quantization Configuration
- Define `QuantizationConfig` class with:
  - `WeightQuantization` (QuantizationMode): Mode for weight quantization
  - `ActivationQuantization` (QuantizationMode): Mode for activation quantization
  - `CalibrationMethod` (CalibrationMethod): Calibration strategy
  - `CalibrationBatchSize` (int): Batch size for calibration runs
  - `QuantizationType` (QuantizationType): Bit-width and signed/unsigned
  - `FallbackToFP32` (bool): Whether to fallback to FP32 for sensitive layers
  - `Validate()`: Method to validate configuration

## File Structure
```
src/
  MLFramework/
    Quantization/
      DataStructures/
        QuantizationMode.cs
        CalibrationMethod.cs
        QuantizationType.cs
        QuantizationParameters.cs
        QuantizationConfig.cs
```

## Implementation Notes
- Use record types where appropriate for immutable structures
- Implement proper validation in QuantizationConfig
- Use C# 12+ features (primary constructors, etc.)

## Dependencies
- None (core data structures)
