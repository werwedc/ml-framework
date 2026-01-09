# Spec: Progressive Model Loading

## Overview
Implement lazy loading mechanism for large models, loading weights on-demand during forward passes.

## Requirements

### 1. ProgressiveModelLoader Class
Manages progressive loading:
- `LoadProgressive(string modelName, string version = null, Device device = null)`: Create model with progressive loading
- `LoadProgressive(ModelMetadata metadata, Device device = null)`: Create model from metadata
- `GetLoadingProgress()`: Get current loading progress (0-1)
- `IsFullyLoaded()`: Check if all weights are loaded
- `PrefetchLayer(string layerName)`: Prefetch specific layer
- `PrefetchLayers(string[] layerNames)`: Prefetch multiple layers

### 2. LayerLoadingStrategy Enum
Define loading strategies:
- OnDemand: Load layer when first accessed
- Sequential: Load layers sequentially during forward pass
- Parallel: Load multiple layers concurrently
- Prefetch: Load next layers while processing current layer

### 3. LazyParameter Class
Parameter that loads weights on demand:
- Inherits from Parameter
- Loads weights from file on first access
- Caches weights after loading
- Supports prefetching
- Tracks loading status

### 4. LayerLoadOrder Class
Define optimal loading order:
- For CNNs: Load from input to output
- For Transformers: Load embeddings first, then attention layers
- For RNNs: Load embedding, then recurrent layers
- Configurable per architecture

### 5. ProgressiveLoadContext
Context for progressive loading:
- ModelMetadata: Metadata for the model
- Device: Target device
- CachePath: Path to cached model file
- LoadingStrategy: Strategy to use
- LoadedLayers: Set of loaded layer names
- LayerLoadOrder: Load order for layers
- LoadingProgress: Current progress (0-1)

### 6. MemoryManager
Manage memory during progressive loading:
- Track memory usage
- Unload unused layers (LRU)
- Limit concurrent loaded layers
- Free memory when fully loaded

### 7. ProgressiveLoadOptions
Configuration options:
- Strategy (LayerLoadingStrategy): Loading strategy (default: OnDemand)
- MaxConcurrentLoads (int): Max layers to load concurrently (default: 3)
- PrefetchCount (int): Number of layers to prefetch (default: 1)
- UnloadStrategy (UnloadStrategy): When to unload layers (Never, MemoryPressure, LRU)
- MaxLoadedLayers (int): Maximum layers to keep in memory (default: -1, unlimited)

### 8. Extension Methods for ModelZoo
Add progressive loading to ModelZoo:
- `LoadProgressive(string modelName, string version = null, Device device = null, ProgressiveLoadOptions options = null)`: Load progressively
- `LoadProgressive(string modelName, LayerLoadingStrategy strategy)`: Load with specific strategy

### 9. Progress Callbacks
Report loading progress:
- OnLayerLoaded event: Fired when a layer finishes loading
- OnProgressChanged event: Fired when overall progress changes
- OnFullyLoaded event: Fired when all layers are loaded

### 10. Unit Tests
Test cases for:
- Load model progressively
- Access lazy parameters trigger loading
- Prefetch layers before access
- Different loading strategies
- Memory management (unloading, limits)
- Progress reporting
- OnDemand vs Sequential vs Parallel strategies
- Edge cases (small models, single layer, access in random order)

## Files to Create
- `src/ModelZoo/Progressive/ProgressiveModelLoader.cs`
- `src/ModelZoo/Progressive/LazyParameter.cs`
- `src/ModelZoo/Progressive/LayerLoadingStrategy.cs`
- `src/ModelZoo/Progressive/LayerLoadOrder.cs`
- `src/ModelZoo/Progressive/ProgressiveLoadContext.cs`
- `src/ModelZoo/Progressive/MemoryManager.cs`
- `src/ModelZoo/Progressive/ProgressiveLoadOptions.cs`
- `tests/ModelZooTests/Progressive/ProgressiveLoadingTests.cs`

## Dependencies
- `ModelMetadata` (from spec_model_metadata.md)
- `ModelZoo` (from spec_model_zoo_load_api.md)
- `ModelCacheManager` (from spec_model_cache_manager.md)
- Existing Parameter class

## Success Criteria
- Can load 10GB+ models without loading entire model into memory
- Forward passes work seamlessly with lazy parameters
- Prefetching improves performance
- Memory limits are respected
- Test coverage > 85%
