# Spec: Accuracy Evaluation and Comparison

## Overview
Implement tools for evaluating and comparing accuracy between FP32 and quantized models.

## Requirements

### 1. Accuracy Metrics
- Implement standard accuracy metrics:
  - `TopKAccuracy` (k=1, 5): Top-K classification accuracy
  - `MAE` (Mean Absolute Error): Regression error
  - `RMSE` (Root Mean Square Error): Regression error
  - `CrossEntropyLoss`: Classification loss

### 2. ModelEvaluator
- Define `ModelEvaluator` class:
  - `Evaluate(IModel model, IDataLoader testData, IMetric metric)`: Evaluate model on test data
  - `EvaluateBatch(IModel model, Tensor<float> batch, Tensor<float> labels, IMetric metric)`: Evaluate single batch
  - `GetPredictions(IModel model, IDataLoader data)`: Get model predictions

### 3. AccuracyComparison
- Define `AccuracyComparison` class:
  - `CompareModels(IModel fp32Model, IModel quantizedModel, IDataLoader testData)`: Compare two models
  - `GetAccuracyDelta()`: Return accuracy difference
  - `GetPerLayerMetrics()`: Return metrics per layer (if available)

### 4. PerLayerSensitivityAnalyzer
- Implement `PerLayerSensitivityAnalyzer`:
  - `AnalyzeLayerSensitivity(IModel model, IDataLoader testData, string[] layerNames)`: Analyze each layer
  - `GetSensitiveLayers(float threshold)`: Return layers exceeding threshold
  - `GetLayerImpactReport()`: Generate impact report per layer

### 5. SensitivityAnalysisResult
- Define `SensitivityAnalysisResult` struct:
  - `LayerName` (string): Name of layer
  - `AccuracyImpact` (float): Accuracy impact when quantized
  - `IsSensitive` (bool): Whether layer is sensitive
  - `RecommendedAction` (string): Recommended action (quantize/fallback)

### 6. AccuracyReport
- Define `AccuracyReport` class:
  - `FP32Accuracy` (float): Baseline accuracy
  - `QuantizedAccuracy` (float): Quantized model accuracy
  - `AccuracyDrop` (float): Difference
  - `IsAcceptable` (bool): Whether drop is acceptable (< configurable threshold)
  - `PerLayerResults` (SensitivityAnalysisResult[]): Per-layer breakdown

### 7. RegressionDetector
- Implement `RegressionDetector`:
  - `DetectRegression(AccuracyReport current, AccuracyReport baseline)`: Detect accuracy regression
  - `GetRegressionDelta()`: Return regression amount
  - `GenerateRegressionReport()`: Generate detailed report

### 8. AccuracyReporter
- Implement `AccuracyReporter`:
  - `GenerateTextReport(AccuracyReport report)`: Generate human-readable report
  - `GenerateJSONReport(AccuracyReport report)`: Generate JSON report
  - `SaveReport(AccuracyReport report, string path)`: Save report to file

## File Structure
```
src/
  MLFramework/
    Quantization/
      Evaluation/
        Metrics/
          IAccuracyMetric.cs
          TopKAccuracy.cs
          MAE.cs
          RMSE.cs
          CrossEntropyLoss.cs
        ModelEvaluator.cs
        AccuracyComparison.cs
        PerLayerSensitivityAnalyzer.cs
        AccuracyReport.cs
        RegressionDetector.cs
        AccuracyReporter.cs
```

## Implementation Notes
- Support batch evaluation for large datasets
- Use parallel evaluation when possible
- Store intermediate results for debugging
- Support multiple metrics simultaneously

## Dependencies
- Core quantization data structures (spec_quantization_data_structures.md)
- Model inference infrastructure
- Tensor operations
- System.Text.Json for JSON reporting
