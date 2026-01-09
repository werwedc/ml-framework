# Spec: Model Registry

## Overview
Implement an in-memory registry to manage ModelMetadata objects and provide lookup functionality.

## Requirements

### 1. ModelRegistry Class
Create a registry with the following operations:
- `Register(ModelMetadata metadata)`: Add a model to the registry
- `Get(string name, string version = null)`: Retrieve metadata by name (and optional version)
- `GetLatestVersion(string name)`: Get the latest version of a model
- `ListAll()`: Return all registered models
- `ListByArchitecture(string architecture)`: Filter by architecture type
- `ListByTask(TaskType task)`: Filter by task type
- `Exists(string name, string version = null)`: Check if a model is registered
- `Remove(string name, string version = null)`: Remove a model from registry

### 2. Registry Storage
- Use in-memory dictionary for fast lookups
- Composite key: {name}_{version}
- Support loading initial registry from JSON files
- Support saving current registry to JSON files

### 3. RegistryBuilder
Helper class to build registries:
- `AddFromDirectory(string path)`: Load all metadata JSON files from a directory
- `AddFromJsonFile(string path)`: Load from a single JSON file
- `AddFromEmbeddedResource(string resource)`: Load from embedded resource (for default models)
- `Build()`: Construct the final registry

### 4. TaskType Enum
Define common ML tasks:
- ImageClassification
- ObjectDetection
- SemanticSegmentation
- TextClassification
- SequenceLabeling
- QuestionAnswering
- TextGeneration
- Regression

### 5. Unit Tests
Test cases for:
- Register and retrieve models
- Version resolution (latest version selection)
- Filtering by architecture and task
- Duplicate registration handling (should throw or update)
- Loading from JSON files
- Registry persistence (save/load)

## Files to Create
- `src/ModelZoo/ModelRegistry.cs`
- `src/ModelZoo/RegistryBuilder.cs`
- `src/ModelZoo/TaskType.cs`
- `tests/ModelZooTests/ModelRegistryTests.cs`

## Dependencies
- `ModelMetadata` from spec_model_metadata.md

## Success Criteria
- Can register 1000+ models with sub-millisecond lookup times
- Version selection correctly resolves to latest
- Registry can be saved and loaded from disk
- Thread-safe operations (for concurrent access)
- Test coverage > 90%
