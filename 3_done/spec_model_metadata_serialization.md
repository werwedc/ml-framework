# Spec: Model Metadata Serialization

## Purpose
Implement JSON serialization/deserialization for model metadata to support persistent storage and artifact management.

## Technical Requirements

### Core Functionality
- Serialize ModelMetadata to JSON
- Deserialize JSON to ModelMetadata
- Support nested dictionaries for hyperparameters and metrics
- Handle DateTime formats (ISO 8601)
- Validate required fields on deserialization

### JSON Schema
```json
{
  "version": "string (required, semantic version)",
  "trainingDate": "ISO 8601 datetime (required)",
  "hyperparameters": {
    "key1": "value (various types)",
    "key2": 123
  },
  "performanceMetrics": {
    "accuracy": 0.95,
    "f1Score": 0.93
  },
  "artifactPath": "string (required, file path or URI)"
}
```

### Data Structures
```csharp
public interface IMetadataSerializer
{
    string Serialize(ModelMetadata metadata);
    ModelMetadata Deserialize(string json);
    void SaveToFile(string filePath, ModelMetadata metadata);
    ModelMetadata LoadFromFile(string filePath);
    Task<string> SerializeAsync(ModelMetadata metadata);
    Task<ModelMetadata> DeserializeAsync(string json);
}
```

## Dependencies
- `spec_model_version_registry.md` (requires ModelMetadata class)

## Testing Requirements
- Serialize simple metadata, verify valid JSON
- Deserialize JSON, verify all fields restored
- Serialize with nested dictionaries, verify structure preserved
- Serialize with various value types (int, float, string, bool)
- Deserialize with missing required fields (should throw)
- Round-trip test (serialize -> deserialize -> compare)
- Async serialization/deserialization tests

## Success Criteria
- [ ] Round-trip serialization preserves all data
- [ ] Handles complex nested dictionary structures
- [ ] DateTime serialization uses ISO 8601 format
- [ ] Validates required fields on load
- [ ] File I/O operations properly handle paths
- [ ] Async methods don't block

## Implementation Notes
- Use `System.Text.Json` for JSON operations (preferred over Newtonsoft.Json)
- Implement custom converters for Dictionary<string, object> to handle mixed types
- Add JsonPropertyName attributes for clarity
- Consider adding schema validation (optional)

## Performance Targets
- Serialize/deserialize single metadata: < 1ms
- Batch serialize 1000 metadata objects: < 100ms
