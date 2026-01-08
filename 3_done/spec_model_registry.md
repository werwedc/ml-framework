# Spec: Model Registry Implementation

## Overview
Implement IModelRegistry interface for registering, tagging, and querying model versions with metadata persistence.

## Tasks

### 1. Create IModelRegistry Interface
**File:** `src/ModelVersioning/IModelRegistry.cs`

```csharp
public interface IModelRegistry
{
    string RegisterModel(string modelPath, ModelMetadata metadata);
    void TagModel(string modelId, string versionTag);
    ModelInfo GetModel(string versionTag);
    ModelInfo GetModelById(string modelId);
    IEnumerable<ModelInfo> ListModels();
    void UpdateModelState(string modelId, LifecycleState newState);
    void SetParentModel(string modelId, string parentModelId);
}
```

### 2. Implement ModelRegistry Class
**File:** `src/ModelVersioning/ModelRegistry.cs`

```csharp
public class ModelRegistry : IModelRegistry
{
    private readonly Dictionary<string, ModelInfo> _modelsById;
    private readonly Dictionary<string, ModelInfo> _modelsByVersion;
    // Constructor with DI support
    // Implement all interface methods
}
```

### 3. Implement Core Methods

#### RegisterModel
- Generate unique model ID (GUID)
- Create ModelInfo with initial LifecycleState.Draft
- Store in both dictionaries (by ID and by version tag if provided)
- Return model ID

#### TagModel
- Associate version tag with model ID
- Validate version tag format (semantic versioning: v1.2.3)
- Update version mappings
- Throw exceptions if version tag already exists

#### GetModel/GetModelById
- Retrieve model info by version tag or ID
- Return null if not found
- Include all metadata

#### ListModels
- Return all registered models
- Support filtering by state (optional parameter)
- Order by creation timestamp

#### UpdateModelState
- Validate state transitions (e.g., Draft -> Staging -> Production)
- Prevent transitions to/from Archived
- Update model state

#### SetParentModel
- Establish parent-child relationship
- Validate parent exists
- Track fine-tuning lineage

## Validation
- Version tag must match semantic versioning pattern
- State transitions must follow allowed transitions
- Parent model must exist
- Duplicate version tags not allowed

## Testing
**File:** `tests/ModelVersioning/ModelRegistryTests.cs`

Create unit tests for:
1. RegisterModel with valid metadata
2. TagModel with valid/invalid version tags
3. GetModel by version tag
4. GetModelById
5. ListModels with and without filters
6. Valid state transitions
7. Invalid state transitions (should throw)
8. Parent-child relationships
9. Duplicate tag prevention
10. Concurrent registration handling

## Dependencies
- Spec: spec_model_data_models.md (must be completed first)
- System.Text.Json
- Regex for version validation
