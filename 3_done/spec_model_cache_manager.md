# Spec: Model Cache Manager

## Overview
Implement a local cache system to store downloaded models with efficient management capabilities.

## Requirements

### 1. ModelCacheManager Class
Cache operations:
- `GetCachePath()`: Get the cache root directory path
- `GetModelPath(string modelName, string version)`: Get full path to cached model file
- `CacheExists(string modelName, string version)`: Check if model is in cache
- `GetCacheSize()`: Return total size of cache in bytes
- `GetCacheSizeFormatted()`: Return human-readable size (e.g., "2.3 GB")
- `ClearCache()`: Delete all cached models
- `RemoveFromCache(string modelName, string version)`: Remove specific model
- `ListCachedModels()`: Return list of all cached models with metadata
- `PruneOldModels(TimeSpan maxAge)`: Remove models older than specified age
- `EnforceMaxCacheSize(long maxBytes)`: Remove least recently used models to fit size limit

### 2. Cache Directory Structure
Organize cache as:
```
~/.ml-framework/model-zoo/
├── registry/
│   └── models.json
└── models/
    ├── resnet50/
    │   ├── v1.0.0/
    │   │   ├── model.bin
    │   │   └── metadata.json
    │   └── v1.1.0/
    │       └── model.bin
    └── bert-base/
        └── v1.2.0/
            └── model.bin
```

### 3. CacheMetadata Class
Store cache tracking information:
- LastAccessed (DateTime): Last time model was accessed
- DownloadDate (DateTime): When model was downloaded
- FileSize (long): Size of cached file
- AccessCount (int): Number of times model was loaded

### 4. Cache Configuration
Configurable settings:
- Default cache location (OS-specific: ~/.ml-framework, %APPDATA%, etc.)
- Maximum cache size (default: 10GB)
- Maximum file age (default: 30 days)
- Cleanup schedule (default: run on startup)

### 5. Cache Statistics
Track and report:
- Total number of cached models
- Total cache size
- Cache hit rate (loads from cache / total loads)
- Least recently used models

### 6. Unit Tests
Test cases for:
- Cache directory creation and path resolution
- Model storage and retrieval
- Cache size calculation and formatting
- Prune old models
- Enforce max cache size (LRU eviction)
- Cache metadata persistence
- Multi-process cache safety (file locking)
- Clear cache functionality

## Files to Create
- `src/ModelZoo/ModelCacheManager.cs`
- `src/ModelZoo/CacheMetadata.cs`
- `src/ModelZoo/CacheConfiguration.cs`
- `src/ModelZoo/CacheStatistics.cs`
- `tests/ModelZooTests/ModelCacheManagerTests.cs`

## Dependencies
- System.IO (for file operations)
- System.Text.Json (for metadata persistence)

## Success Criteria
- Cache correctly stores and retrieves models
- LRU eviction removes correct models
- Cache size calculations are accurate
- Multi-process access is safe
- Test coverage > 90%
