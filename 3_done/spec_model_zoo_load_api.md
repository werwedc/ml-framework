# Spec: Basic Model Zoo Load API

## Overview
Implement the primary ModelZoo API for loading pre-trained models with caching support.

## Requirements

### 1. ModelZoo Static Class
Core loading methods:
- `Load(string modelName, string version = null, string variant = null, bool pretrained = true, Device device = null)`: Load a model
- `Load<T>(string modelName, string version = null, string variant = null, bool pretrained = true, Device device = null)`: Generic load with type specification
- `LoadFromPath(string path, Device device = null)`: Load a model from a local file path
- `IsAvailable(string modelName, string version = null)`: Check if model exists in registry
- `ListModels()`: List all available models

### 2. LoadFlow Class
Internal class to orchestrate the loading process:
- Check registry for model metadata
- Verify cache for existing model
- Download model if not in cache (using ModelDownloadService)
- Deserialize model from disk
- Move model to specified device
- Apply pre-trained weights if requested
- Update cache metadata (access time, count)

### 3. Model Deserialization
Support deserializing model weights:
- Read native format files
- Map weights to model layers using layer naming conventions
- Handle weight shape mismatches gracefully (throw descriptive exceptions)
- Support partial loading (e.g., only specific layers)

### 4. Error Handling
Define exceptions:
- `ModelNotFoundException`: Model not in registry
- `VersionNotFoundException`: Specified version doesn't exist
- `DownloadFailedException`: Download failed (rethrow from download service)
- `DeserializationException`: Model file is corrupted or invalid format
- `IncompatibleModelException`: Model architecture doesn't match expected structure

### 5. Configuration
ModelZoo settings:
- Default device (CPU/GPU)
- Cache enabled/disabled
- Auto-download enabled/disabled
- Default download timeout

### 6. Integration Points
Connect to:
- ModelRegistry (for metadata lookup)
- ModelDownloadService (for downloads)
- ModelCacheManager (for cache operations)
- ModelDeserializer (for loading weights)

### 7. Unit Tests
Test cases for:
- Load model from cache
- Load model with download
- Load specific version
- Load with pretrained weights
- Load without pretrained weights (random initialization)
- Error cases (not found, download fails, corrupt file)
- Device handling (CPU, GPU)
- Generic type specification

## Files to Create
- `src/ModelZoo/ModelZoo.cs`
- `src/ModelZoo/LoadFlow.cs`
- `src/ModelZoo/ModelDeserializer.cs`
- `src/ModelZoo/ModelZooConfiguration.cs`
- `src/ModelZoo/Exceptions/ModelNotFoundException.cs`
- `src/ModelZoo/Exceptions/VersionNotFoundException.cs`
- `src/ModelZoo/Exceptions/IncompatibleModelException.cs`
- `tests/ModelZooTests/ModelZooTests.cs`

## Dependencies
- All previous ModelZoo specs (Metadata, Registry, DownloadService, CacheManager)
- Existing Model/Device/Parameter infrastructure in the framework

## Success Criteria
- Can load ResNet-50 from ImageNet successfully
- First load downloads, second load uses cache
- All error cases throw appropriate exceptions
- Test coverage > 85%
