# Spec: Model Recommendation Engine

## Overview
Implement an intelligent recommendation system to suggest models based on input characteristics and task constraints.

## Requirements

### 1. ModelConstraints Class
Define constraints for recommendation:
- InputShape (Shape): Input tensor shape
- Task (TaskType): Target task
- MaxLatency (float): Maximum inference latency in ms (optional)
- MaxMemory (long): Maximum memory usage in bytes (optional)
- MinAccuracy (float): Minimum required accuracy (optional)
- MaxFileSize (long): Maximum model file size (optional)
- Device (DeviceType): Target device (CPU, GPU, Edge) (optional)
- BatchSize (int): Expected batch size (optional)
- DeploymentEnvironment (DeploymentEnv): Cloud, Edge, Mobile, etc. (optional)

### 2. ModelRecommendation Class
Recommendation result:
- Model (ModelMetadata): Recommended model
- Reason (string): Explanation for recommendation
- EstimatedLatency (float): Estimated inference latency
- EstimatedMemory (float): Estimated memory usage
- CompatibilityScore (double): How well it fits constraints (0-1)
- Alternatives (List<ModelMetadata>): Alternative recommendations

### 3. ModelRecommendationEngine
Recommendation operations:
- `RecommendFor(Shape inputShape, TaskType task, ModelConstraints constraints = null)`: Get single best recommendation
- `RecommendMultiple(Shape inputShape, TaskType task, int topN, ModelConstraints constraints = null)`: Get top N recommendations
- `RecommendForConstraints(ModelConstraints constraints)`: Get recommendations based on constraints only
- `GetAlternatives(string modelName, int count)`: Get alternatives to a specific model

### 4. Compatibility Scoring
Score models based on constraints:
- Input shape compatibility: exact match = 1.0, compatible dimensions = 0.8, incompatible = 0.0
- Latency constraint: (MaxLatency - EstimatedLatency) / MaxLatency (capped at 0-1)
- Memory constraint: (MaxMemory - EstimatedMemory) / MaxMemory (capped at 0-1)
- Accuracy constraint: ActualAccuracy / MinAccuracy (capped at 1.0)
- File size constraint: (MaxFileSize - FileSize) / MaxFileSize (capped at 0-1)
- Overall score: Weighted average of individual scores

### 5. Latency Estimator
Estimate model inference latency:
- `EstimateLatency(ModelMetadata model, DeviceType device, int batchSize = 1)`: Estimate inference time
- Use parameter count as proxy: Latency ~ Parameters / DeviceFlops
- Architecture-specific adjustments (CNNs vs Transformers)
- Device-specific factors (CPU vs GPU vs TPU)

### 6. Memory Estimator
Estimate model memory usage:
- `EstimateMemory(ModelMetadata model, int batchSize = 1)`: Estimate memory required
- Calculate: ModelWeightsMemory + ActivationMemory
- ActivationMemory ~ BatchSize * InputShape * OutputDimensions * Layers
- Device-specific memory overhead

### 7. Recommendation Strategies
Different strategies for different use cases:
- `AccuracyFirst`: Prioritize highest accuracy
- `PerformanceFirst`: Prioritize lowest latency
- `Balanced`: Balance accuracy and performance
- `MemoryConstrained`: Prioritize smallest models
- `EdgeDeployment`: Prioritize models suitable for edge devices

### 8. Unit Tests
Test cases for:
- Recommend based on input shape
- Recommend with latency constraints
- Recommend with memory constraints
- Recommend with accuracy constraints
- Get multiple recommendations
- Get alternatives to specific model
- Compatibility scoring
- Latency estimation (sanity check: larger models = slower)
- Memory estimation
- Different recommendation strategies
- Edge cases (no matching models, all constraints met)

## Files to Create
- `src/ModelZoo/Discovery/ModelConstraints.cs`
- `src/ModelZoo/Discovery/ModelRecommendation.cs`
- `src/ModelZoo/Discovery/ModelRecommendationEngine.cs`
- `src/ModelZoo/Discovery/LatencyEstimator.cs`
- `src/ModelZoo/Discovery/MemoryEstimator.cs`
- `src/ModelZoo/Discovery/RecommendationStrategy.cs`
- `tests/ModelZooTests/Discovery/RecommendationTests.cs`

## Dependencies
- `ModelRegistry` (from spec_model_registry.md)
- `ModelMetadata` (from spec_model_metadata.md)
- `TaskType` (from spec_model_registry.md)
- Existing Shape class (if available)

## Success Criteria
- Returns sensible recommendations for common tasks
- Respects all constraints
- Provides reasonable latency/memory estimates
- Alternative models are relevant
- Test coverage > 85%
