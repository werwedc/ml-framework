# Spec: External Model Hub Integration

## Overview
Implement integration with external model repositories (Hugging Face, TensorFlow Hub, ONNX Model Zoo) through a unified interface.

## Requirements

### 1. IModelHub Interface
Base interface for model hubs:
- `string HubName`: Hub identifier (e.g., "huggingface", "tensorflow", "onnx")
- `Task<ModelMetadata> GetModelMetadataAsync(string modelId)`: Get model metadata
- `Task<Stream> DownloadModelAsync(string modelId, IProgress<double> progress = null)`: Download model
- `Task<bool> ModelExistsAsync(string modelId)`: Check if model exists
- `Task<string[]> ListModelsAsync(string filter = null)`: List available models
- `bool CanHandle(string modelId)`: Check if hub can handle a model ID

### 2. ModelIdParser
Parse model identifiers:
- `Parse(string modelId)`: Parse model ID string (e.g., "hub:huggingface/bert-base-uncased")
- Returns: HubName, ModelName, Version, Variant

Supported formats:
- `bert-base` (local registry)
- `hub:huggingface/bert-base-uncased`
- `hub:tensorflow/resnet50-v2`
- `hub:onnx/vgg16-7`
- `custom:my-registry/efficientnet`

### 3. HuggingFaceHub
Hugging Face Hub implementation:
- Base URL: https://huggingface.co
- Support for model repository format
- Parse README.md for metadata
- Download model files (safetensors, pytorch_model.bin, etc.)
- Handle different file formats

### 4. TensorFlowHub
TensorFlow Hub implementation:
- Base URL: https://tfhub.dev
- Support for TF Hub model format
- Download SavedModel or HDF5 files
- Parse model metadata from hub manifest

### 5. ONNXHub
ONNX Model Zoo implementation:
- Base URL: https://github.com/onnx/models
- Support for ONNX format
- Download .onnx files
- Parse model metadata from model zoo repository

### 6. HubRegistry
Register and manage hubs:
- `RegisterHub(IModelHub hub)`: Register a hub
- `UnregisterHub(string hubName)`: Unregister a hub
- `GetHub(string hubName)`: Get registered hub
- `ListHubs()`: List all registered hubs
- `GetDefaultHub()`: Get default hub for non-prefixed model IDs

### 7. Unified ModelZoo Extensions
Extend ModelZoo to support external hubs:
- `Load(string modelId, bool pretrained = true, Device device = null)`: Parse ID and delegate to appropriate hub
- `GetModelMetadata(string modelId)`: Get metadata from appropriate hub
- `ListHubModels(string hubName, string filter = null)`: List models from specific hub

### 8. Authentication Support
Interface for hub authentication:
- `IHubAuthentication`: Base authentication interface
- `ApiKeyAuth`: API key-based authentication
- `TokenAuth`: OAuth token authentication
- `AnonymousAuth`: No authentication required
- Hubs can specify required auth method

### 9. HubConfiguration
Configuration for hubs:
- `HuggingFaceHub`: API token, default repo, timeout
- `TensorFlowHub`: timeout, compression
- `ONNXHub`: timeout, mirror URLs

### 10. Unit Tests
Test cases for:
- Model ID parsing (various formats)
- Register and unregister hubs
- Get hub by name
- HuggingFace hub metadata retrieval (mock)
- TensorFlow hub metadata retrieval (mock)
- ONNX hub metadata retrieval (mock)
- Authentication configuration
- Load from external hub (mock)
- Error cases (hub not found, model not found, invalid ID)

## Files to Create
- `src/ModelZoo/ExternalHubs/IModelHub.cs`
- `src/ModelZoo/ExternalHubs/ModelIdParser.cs`
- `src/ModelZoo/ExternalHubs/HuggingFaceHub.cs`
- `src/ModelZoo/ExternalHubs/TensorFlowHub.cs`
- `src/ModelZoo/ExternalHubs/ONNXHub.cs`
- `src/ModelZoo/ExternalHubs/HubRegistry.cs`
- `src/ModelZoo/ExternalHubs/IHubAuthentication.cs`
- `src/ModelZoo/ExternalHubs/ApiKeyAuth.cs`
- `src/ModelZoo/ExternalHubs/HubConfiguration.cs`
- `tests/ModelZooTests/ExternalHubs/HubTests.cs`

## Dependencies
- `ModelMetadata` (from spec_model_metadata.md)
- `ModelZoo` (from spec_model_zoo_load_api.md)
- System.Net.Http (for HTTP requests)

## Success Criteria
- Can parse all supported model ID formats
- Can register and use multiple hubs
- Hubs correctly handle model metadata
- Authentication works as expected
- Test coverage > 85%
