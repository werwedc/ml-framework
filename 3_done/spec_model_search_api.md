# Spec: Model Search API

## Overview
Implement search functionality to query the model registry by various criteria.

## Requirements

### 1. ModelSearchQuery Class
Define query parameters:
- Task (TaskType): Filter by task type (optional)
- Architecture (string): Filter by architecture (optional)
- MinAccuracy (double): Minimum accuracy threshold (optional)
- MaxAccuracy (double): Maximum accuracy threshold (optional)
- MaxFileSize (long): Maximum file size in bytes (optional)
- MinFileSize (long): Minimum file size in bytes (optional)
- MaxParameters (long): Maximum number of parameters (optional)
- MinParameters (long): Minimum number of parameters (optional)
- License (string): Filter by license (optional)
- InputShape (int[]): Filter by compatible input shape (optional)
- OutputShape (int[]): Filter by compatible output shape (optional)
- FrameworkVersion (string): Filter by framework version compatibility (optional)
- PretrainedDataset (string): Filter by training dataset (optional)
- CustomFilters (Dictionary<string, Func<ModelMetadata, bool>>): Custom filter functions (optional)
- SortBy (SearchSortBy): Sort criteria (Name, Accuracy, Size, Parameters, Date)
- SortDescending (bool): Sort order (default: descending)
- Limit (int): Maximum number of results (default: 100)

### 2. ModelSearchResult Class
Result metadata:
- Model (ModelMetadata): The model metadata
- MatchScore (double): How well it matches the query (0-1)
- MatchReasons (string[]): Reasons why it matched

### 3. ModelSearchService
Search operations:
- `Search(ModelSearchQuery query)`: Search registry with query
- `SearchByTask(TaskType task, int limit = 10)`: Quick search by task
- `SearchByAccuracy(double minAccuracy, TaskType task, int limit = 10)`: Search by accuracy
- `SearchByArchitecture(string architecture, int limit = 10)`: Search by architecture type
- `SearchBySize(long maxSize, TaskType task, int limit = 10)`: Search by file size
- `AdvancedSearch(Action<ModelSearchQueryBuilder> build)`: Fluent query builder

### 4. ModelSearchQueryBuilder
Fluent API for building queries:
- `WithTask(TaskType task)`: Set task filter
- `WithArchitecture(string architecture)`: Set architecture filter
- `WithAccuracyRange(double min, double max)`: Set accuracy range
- `WithFileSizeRange(long min, long max)`: Set file size range
- `WithLicense(string license)`: Set license filter
- `WithInputShape(int[] shape)`: Set input shape filter
- `SortBy(SearchSortBy sortBy)`: Set sort criteria
- `WithLimit(int limit)`: Set result limit
- `AddCustomFilter(string name, Func<ModelMetadata, bool> filter)`: Add custom filter
- `Build()`: Build the query object

### 5. SearchSortBy Enum
Sort options:
- Name: Sort by model name
- Accuracy: Sort by accuracy metric
- Size: Sort by file size
- Parameters: Sort by number of parameters
- Date: Sort by release date (if available)

### 6. Scoring Algorithm
Calculate match scores based on:
- Exact matches on required fields (score = 1.0)
- Close matches on continuous values (accuracy, size) using normalized distance
- Partial matches on string fields (architecture name)
- Custom filter weights

### 7. Unit Tests
Test cases for:
- Search by task
- Search by accuracy range
- Search by architecture
- Complex queries with multiple filters
- Custom filters
- Sorting (asc/desc)
- Limit results
- Match score calculation
- Query builder API
- Edge cases (no results, all models match)

## Files to Create
- `src/ModelZoo/Discovery/ModelSearchQuery.cs`
- `src/ModelZoo/Discovery/ModelSearchResult.cs`
- `src/ModelZoo/Discovery/ModelSearchService.cs`
- `src/ModelZoo/Discovery/ModelSearchQueryBuilder.cs`
- `src/ModelZoo/Discovery/SearchSortBy.cs`
- `tests/ModelZooTests/Discovery/SearchTests.cs`

## Dependencies
- `ModelRegistry` (from spec_model_registry.md)
- `ModelMetadata` (from spec_model_metadata.md)
- `TaskType` (from spec_model_registry.md)

## Success Criteria
- Can search 1000+ models in < 100ms
- Accurate match scoring
- Fluent query builder works correctly
- Test coverage > 90%
