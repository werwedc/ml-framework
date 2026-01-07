# Spec: Post-Training Quantization API

## Overview
Implement main API for post-training quantization (PTQ) with support for dynamic and static quantization.

## Requirements

### 1. IQuantizer Interface
- Define `IQuantizer` interface with:
  - `Quantize(IModel model, IDataLoader calibrationData, QuantizationConfig config)`: Quantize model
  - `SetPerLayerFallback(string layerName, bool enabled)`: Enable/disable layer quantization

### 2. PTQ Quantizer
- Implement `PTQQuantizer` class:
  - Supports both dynamic and static quantization
  - Calibration with configurable calibrator
  - Per-layer quantization control

### 3. Dynamic Quantization
- Implement dynamic quantization logic:
  - Quantize weights to Int8 (static)
  - Keep activations in FP32 (quantized on-the-fly during inference)
  - Ideal for RNNs and memory-constrained inference

### 4. Static Quantization
- Implement static quantization logic:
  - Quantize both weights and activations to Int8
  - Run calibration to determine activation ranges
  - Best for CNNs and feedforward networks

### 5. Model Traversal
- Implement model graph traversal:
  - Visit each layer in computational graph
  - Identify quantizable layers (Linear, Conv2d, etc.)
  - Skip non-quantizable layers (activation functions, pooling)

### 6. Calibration Process
- Implement calibration workflow:
  - Run inference on calibration data
  - Collect statistics for each layer
  - Compute quantization parameters
  - Apply quantization to weights and activations

### 7. Quantization Application
- Implement quantization application:
  - Replace FP32 weights with quantized weights
  - Store quantization parameters per layer
  - Update layer metadata for dequantization

### 8. Sensitivity Analysis
- Implement per-layer sensitivity analysis:
  - Evaluate accuracy impact of quantizing each layer
  - Identify sensitive layers that need FP32 fallback
  - Support automatic fallback for low-sensitivity layers

## File Structure
```
src/
  MLFramework/
    Quantization/
      PTQ/
        IQuantizer.cs
        PTQQuantizer.cs
        DynamicQuantization.cs
        StaticQuantization.cs
        ModelTraversal.cs
        CalibrationProcess.cs
        SensitivityAnalysis.cs
```

## Implementation Notes
- Use existing model graph infrastructure
- Support incremental calibration (batches)
- Preserve original model metadata
- Handle layer dependencies (e.g., activation following conv)

## Dependencies
- Core quantization data structures (spec_quantization_data_structures.md)
- Calibration strategies (spec_calibration_strategies.md)
- Quantization operations (spec_quantization_operations.md)
- Model graph and layer interfaces
