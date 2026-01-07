# Spec: Quantization-Aware Training (QAT) Fake Quantization

## Overview
Implement fake quantization operations that simulate Int8 arithmetic during training.

## Requirements

### 1. FakeQuantize Operation
- Implement `FakeQuantize(Tensor<float> input, QuantizationParameters parameters)`:
  - Simulate quantization noise during forward pass
  - Maintain differentiable gradient for backpropagation
  - Round to nearest integer but keep in FP32 for gradients

### 2. STE (Straight-Through Estimator)
- Implement STE gradient computation:
  - Use identity gradient for backward pass
  - Bypass non-differentiable rounding operation
  - Clip gradients to prevent explosion

### 3. FakeQuantizeLayer
- Define `FakeQuantizeLayer` class:
  - Wrap existing layers with fake quantization
  - Store quantization parameters as learnable parameters
  - Support per-tensor and per-channel quantization

### 4. QAT Module Wrapper
- Implement `QATModuleWrapper`:
  - Wrap standard layers (Linear, Conv2d) with fake quantization
  - Insert fake quant nodes before and after layer operations
  - Preserve original layer behavior

### 5. Moving Average Statistics
- Implement moving average for quantization parameters:
  - Update scale and zero-point during training
  - Smooth out activation distribution changes
  - Configurable momentum parameter

### 6. Observer Pattern
- Implement activation observers:
  - Track min/max statistics during training
  - Update quantization parameters periodically
  - Support different observer strategies (min-max, moving average)

### 7. Backward Compatibility
- Ensure fake quantization layers are:
  - Transparent during inference (use true Int8)
  - Differentiable during training (use STE)
  - Compatible with existing optimizers

## File Structure
```
src/
  MLFramework/
    Quantization/
      QAT/
        FakeQuantizeOperation.cs
        StraightThroughEstimator.cs
        FakeQuantizeLayer.cs
        QATModuleWrapper.cs
        MovingAverageStatistics.cs
        ActivationObserver.cs
```

## Implementation Notes
- Use PyTorch-style fake quantization logic
- Ensure gradients flow correctly through fake quant nodes
- Support both forward and backward hooks
- Maintain quantization parameters as separate tensors

## Dependencies
- Core quantization data structures (spec_quantization_data_structures.md)
- Quantization operations (spec_quantization_operations.md)
- Tensor operations and autograd
