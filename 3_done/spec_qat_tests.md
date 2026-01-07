# Spec: Quantization-Aware Training Integration Tests

## Overview
Implement integration tests for quantization-aware training (QAT) functionality.

## Requirements

### 1. FakeQuantize Tests
- Test fake quantization operation:
  - Verify quantization noise is simulated
  - Verify gradients flow correctly (STE)
  - Check forward pass produces noisy outputs
  - Check backward pass uses identity gradients

### 2. FakeQuantizeLayer Tests
- Test `FakeQuantizeLayer`:
  - Wrap standard layer correctly
  - Store quantization parameters correctly
  - Update parameters during training
  - Support per-tensor and per-channel modes

### 3. QATModuleWrapper Tests
- Test `QATModuleWrapper`:
  - Wrap Linear layers correctly
  - Wrap Conv2D layers correctly
  - Preserve original layer behavior
  - Insert fake quant nodes in correct positions

### 4. Model Preparation Tests
- Test model graph transformation:
  - Prepare simple model for QAT
  - Verify fake quant nodes are inserted
  - Verify model structure is preserved
  - Check quantization parameters are initialized

### 5. Training Integration Tests
- Test QAT training:
  - Train model with fake quantization
  - Verify gradients flow through fake quant nodes
  - Verify weights update correctly
  - Verify quantization parameters evolve

### 6. Moving Average Statistics Tests
- Test moving average behavior:
  - Update statistics during training
  - Verify smoothing works correctly
  - Check momentum parameter behavior
  - Reset statistics correctly

### 7. Activation Observer Tests
- Test activation observers:
  - Track min/max statistics correctly
  - Update quantization parameters
  - Handle different observer strategies
  - Reset correctly

### 8. QATPreparer Tests
- Test `QATPreparer`:
  - Prepare model for QAT
  - Convert trained QAT model to Int8
  - Get QAT statistics
  - Handle per-layer configuration

### 9. Checkpointing Tests
- Test QAT model checkpointing:
  - Save QAT model with quantization parameters
  - Restore QAT model from checkpoint
  - Verify state is preserved correctly
  - Support resume training

### 10. End-to-End QAT Tests
- Test complete QAT workflow:
  - Train simple model with QAT (MNIST)
  - Convert to Int8 after training
  - Compare QAT vs PTQ accuracy
  - Verify QAT achieves better accuracy than PTQ

### 11. Layer-wise Config Tests
- Test per-layer QAT configuration:
  - Override global config per layer
  - Disable QAT for specific layers
  - Mix quantized and non-quantized layers
  - Verify final model correctness

### 12. Conversion Tests
- Test final conversion to Int8:
  - Extract trained quantization parameters
  - Convert weights to Int8
  - Remove fake quantization nodes
  - Verify inference accuracy

## File Structure
```
tests/
  MLFramework.Tests/
    Quantization/
      QAT/
        FakeQuantizeTests.cs
        FakeQuantizeLayerTests.cs
        QATModuleWrapperTests.cs
        ModelPreparationTests.cs
        TrainingIntegrationTests.cs
        MovingAverageStatisticsTests.cs
        ActivationObserverTests.cs
        QATPreparerTests.cs
        CheckpointingTests.cs
        EndToEndQATTests.cs
        LayerwiseConfigTests.cs
        ConversionTests.cs
```

## Implementation Notes
- Use synthetic models for quick tests
- Use real datasets (MNIST) for end-to-end tests
- Monitor gradient flow and parameter updates
- Compare QAT results with PTQ baseline

## Dependencies
- Core quantization components
- QAT API components
- Fake quantization operations
- Model training infrastructure
- Autograd infrastructure
- Checkpoint infrastructure
- xUnit test framework
