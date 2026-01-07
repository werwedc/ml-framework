# Spec: Accuracy Evaluation Tests

## Overview
Implement tests for accuracy evaluation and comparison functionality.

## Requirements

### 1. Accuracy Metrics Tests
- Test accuracy metrics:
  - `TopKAccuracy`: Correctly computes top-k accuracy
  - `MAE`: Correctly computes mean absolute error
  - `RMSE`: Correctly computes root mean square error
  - `CrossEntropyLoss`: Correctly computes cross-entropy

### 2. ModelEvaluator Tests
- Test `ModelEvaluator`:
  - Evaluate model on test data
  - Evaluate single batch
  - Get predictions
  - Handle different batch sizes

### 3. AccuracyComparison Tests
- Test `AccuracyComparison`:
  - Compare FP32 vs quantized model
  - Compute accuracy delta correctly
  - Generate comparison report
  - Handle equal models

### 4. PerLayerSensitivityAnalyzer Tests
- Test per-layer sensitivity:
  - Analyze each layer independently
  - Identify sensitive layers correctly
  - Generate sensitivity reports
  - Test different thresholds

### 5. SensitivityAnalysisResult Tests
- Test `SensitivityAnalysisResult`:
  - Correct data structure
  - Correct field population
  - Correct recommendation logic

### 6. AccuracyReport Tests
- Test `AccuracyReport`:
  - Correctly populate all fields
  - Compute accuracy drop correctly
  - Determine if accuracy is acceptable
  - Aggregate per-layer results

### 7. RegressionDetector Tests
- Test regression detection:
  - Detect regression between two reports
  - Compute regression delta correctly
  - Generate regression report
  - Handle no regression cases

### 8. AccuracyReporter Tests
- Test `AccuracyReporter`:
  - Generate text report correctly
  - Generate JSON report correctly
  - Save report to file
  - Load report from file

### 9. Integration Tests
- Test accuracy evaluation workflow:
  - Train simple model (MNIST)
  - Apply PTQ
  - Evaluate FP32 accuracy
  - Evaluate quantized accuracy
  - Generate comparison report

### 10. Edge Cases Tests
- Test edge cases:
  - Empty test data
  - Single test sample
  - Perfect accuracy (100%)
  - Zero accuracy (0%)
  - NaN/Inf in predictions

## File Structure
```
tests/
  MLFramework.Tests/
    Quantization/
      Evaluation/
        AccuracyMetricsTests.cs
        ModelEvaluatorTests.cs
        AccuracyComparisonTests.cs
        PerLayerSensitivityAnalyzerTests.cs
        AccuracyReportTests.cs
        RegressionDetectorTests.cs
        AccuracyReporterTests.cs
        AccuracyEvaluationIntegrationTests.cs
        EdgeCasesTests.cs
```

## Implementation Notes
- Use synthetic data for quick tests
- Use real models (MNIST classifier) for integration tests
- Use fixed random seeds for reproducibility
- Compare results with expected values (with tolerance)

## Dependencies
- Core quantization components
- PTQ components
- Accuracy evaluation components
- Model training infrastructure
- Dataset loading infrastructure
- xUnit test framework
