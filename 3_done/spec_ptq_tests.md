# Spec: Post-Training Quantization Integration Tests

## Overview
Implement integration tests for post-training quantization (PTQ) functionality.

## Requirements

### 1. Dynamic Quantization Tests
- Test dynamic quantization:
  - Quantize simple Linear model
  - Verify weights are quantized to Int8
  - Verify activations remain FP32
  - Check inference results are accurate

### 2. Static Quantization Tests
- Test static quantization:
  - Quantize simple Conv2D model with calibration
  - Verify weights and activations are quantized
  - Check calibration statistics are collected
  - Verify inference results are accurate

### 3. PTQ Quantizer Tests
- Test `PTQQuantizer`:
  - Quantize multi-layer model
  - Verify all quantizable layers are quantized
  - Verify non-quantizable layers are preserved
  - Check quantization parameters per layer

### 4. Calibration Process Tests
- Test calibration workflow:
  - Run calibration on synthetic data
  - Verify statistics are collected
  - Verify quantization parameters are computed
  - Check calibration convergence

### 5. Model Traversal Tests
- Test model graph traversal:
  - Correct identification of quantizable layers
  - Correct handling of layer dependencies
  - Correct preservation of model structure
  - Handling of complex models (residual connections)

### 6. Per-Layer Fallback Tests
- Test per-layer quantization control:
  - Enable/disable quantization per layer
  - Verify FP32 fallback for sensitive layers
  - Check mixed-precision models work correctly
  - Verify inference with mixed-precision

### 7. Sensitivity Analysis Tests
- Test per-layer sensitivity:
  - Analyze sensitivity for each layer
  - Identify sensitive layers correctly
  - Verify sensitivity thresholds work
  - Generate sensitivity report

### 8. End-to-End PTQ Tests
- Test complete PTQ workflow:
  - Train simple model (MNIST classifier)
  - Apply PTQ with calibration
  - Compare FP32 vs quantized accuracy
  - Verify accuracy drop is acceptable (< 1%)

### 9. Quantization Configuration Tests
- Test different configurations:
  - Per-tensor vs per-channel quantization
  - Symmetric vs asymmetric quantization
  - Different calibration methods
  - Mixed configurations

### 10. Edge Cases Tests
- Test edge cases:
  - Empty calibration data
  - Single batch calibration
  - Very small models
  - Models with no quantizable layers

## File Structure
```
tests/
  MLFramework.Tests/
    Quantization/
      PTQ/
        DynamicQuantizationTests.cs
        StaticQuantizationTests.cs
        PTQQuantizerTests.cs
        CalibrationProcessTests.cs
        ModelTraversalTests.cs
        PerLayerFallbackTests.cs
        SensitivityAnalysisTests.cs
        EndToEndPTQTests.cs
        QuantizationConfigTests.cs
        EdgeCasesTests.cs
```

## Implementation Notes
- Use synthetic models for quick tests
- Use real datasets (MNIST) for end-to-end tests
- Measure execution time for performance tracking
- Use test fixtures for common setup

## Dependencies
- Core quantization components
- PTQ API components
- Model training infrastructure
- Dataset loading infrastructure
- xUnit test framework
