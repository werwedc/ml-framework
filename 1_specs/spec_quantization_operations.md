# Spec: Quantization Operations

## Overview
Implement core quantization operations for converting FP32 to Int8 and vice versa.

## Requirements

### 1. Quantize Operation
- Implement `Quantize(float value, QuantizationParameters parameters)`:
  - Scale and shift value based on parameters
  - Clamp to Int8 range [-128, 127] or UInt8 range [0, 255]
  - Apply rounding (round half away from zero)

### 2. Dequantize Operation
- Implement `Dequantize(sbyte quantizedValue, QuantizationParameters parameters)`:
  - Convert quantized value back to FP32
  - Apply scale and zero-point reversal

### 3. Tensor Quantization (Per-Tensor)
- Implement `QuantizeTensor(float[] tensor, QuantizationParameters parameters)`:
  - Quantize entire tensor with single scale/zero-point
  - Return sbyte[] quantized tensor
  - Support both symmetric and asymmetric modes

### 4. Tensor Dequantization (Per-Tensor)
- Implement `DequantizeTensor(sbyte[] quantizedTensor, QuantizationParameters parameters)`:
  - Dequantize entire tensor to FP32
  - Return float[] dequantized tensor

### 5. Per-Channel Quantization
- Implement `QuantizeTensorPerChannel(float[] tensor, QuantizationParameters[] parameters)`:
  - Quantize tensor with per-channel parameters
  - Support N-dimensional tensors (channels-first layout)
  - Return quantized tensor with per-channel parameters

### 6. Per-Channel Dequantization
- Implement `DequantizeTensorPerChannel(sbyte[] quantizedTensor, QuantizationParameters[] parameters)`:
  - Dequantize with per-channel parameters
  - Return FP32 tensor

### 7. QuantizationUtils Static Class
- Define helper methods:
  - `CalculateScale(float min, float max, int quantMin, int quantMax)`: Compute scale factor
  - `CalculateZeroPoint(float min, float max, float scale, int quantMin, int quantMax)`: Compute zero-point
  - `Clamp(int value, int min, int max)`: Clamp value to range

## File Structure
```
src/
  MLFramework/
    Quantization/
      Operations/
        QuantizationOperations.cs
        DequantizationOperations.cs
        PerChannelOperations.cs
        QuantizationUtils.cs
```

## Implementation Notes
- Use SIMD (Vector256) for performance where possible
- Handle NaN and Inf values appropriately
- Round half away from zero for better accuracy
- Support both sbyte (Int8) and byte (UInt8)

## Dependencies
- Core quantization data structures (spec_quantization_data_structures.md)
- System.Numerics for SIMD operations
