# Spec: Quantization-Aware Training (QAT) API

## Overview
Implement API for preparing and managing quantization-aware training.

## Requirements

### 1. IQATPreparer Interface
- Define `IQATPreparer` interface with:
  - `PrepareForQAT(IModel model, QuantizationConfig config)`: Prepare model for QAT
  - `ConvertToQuantized(IModel qatModel)`: Convert trained QAT model to Int8
  - `GetQATStatistics(IModel qatModel)`: Get quantization statistics

### 2. QATPreparer Implementation
- Implement `QATPreparer` class:
  - Insert fake quantization nodes into model graph
  - Configure quantization parameters per layer
  - Manage QAT training lifecycle

### 3. Model Preparation
- Implement model graph transformation:
  - Identify quantizable layers (Linear, Conv2d)
  - Insert FakeQuantizeLayer before weight operations
  - Insert FakeQuantizeLayer after activation
  - Replace original layers with QAT-wrapped layers

### 4. Layer-wise Quantization Config
- Support per-layer configuration:
  - Override global config for specific layers
  - Set quantization mode per layer
  - Enable/disable quantization per layer

### 5. Training Integration
- Ensure QAT compatibility with:
  - Existing optimizers (no changes needed)
  - Existing training loops (no changes needed)
  - Existing loss functions (no changes needed)

### 6. Quantization Statistics
- Implement statistics collection:
  - Track quantization parameter evolution during training
  - Record activation ranges per layer
  - Monitor gradient flow through fake quant nodes

### 7. Conversion to Quantized Model
- Implement final conversion:
  - Extract trained quantization parameters
  - Convert weights to Int8
  - Remove fake quantization nodes
  - Generate true Int8 model for inference

### 8. Checkpointing
- Support QAT model checkpointing:
  - Save quantization parameters with model weights
  - Restore QAT state from checkpoint
  - Compatible with existing save/load infrastructure

## File Structure
```
src/
  MLFramework/
    Quantization/
      QAT/
        IQATPreparer.cs
        QATPreparer.cs
        ModelPreparation.cs
        LayerwiseConfig.cs
        TrainingIntegration.cs
        QATStatistics.cs
        ModelConversion.cs
        QATCheckpointing.cs
```

## Implementation Notes
- Preserve original model structure for easy conversion
- Support partial QAT (only some layers)
- Ensure conversion produces valid quantized model
- Handle edge cases (frozen layers, batch norm)

## Dependencies
- Core quantization data structures (spec_quantization_data_structures.md)
- Fake quantization operations (spec_qat_fake_quantization.md)
- Model graph transformation utilities
- Checkpoint/save-load infrastructure
