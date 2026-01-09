# Spec: Model Metadata System

## Overview
Define the metadata schema and data structures for the Model Zoo registry.

## Requirements

### 1. ModelMetadata Class
Create a C# class to represent model metadata with the following properties:
- Name (string): Model name/identifier
- Version (string): Semantic version (e.g., "1.0.0")
- Architecture (string): Architecture type (e.g., "ResNet", "BERT")
- Variants (string[]): Available variants (e.g., ["resnet18", "resnet50"])
- PretrainedOn (string): Dataset used for pre-training
- PerformanceMetrics (Dictionary<string, double>): Metrics like accuracy, F1-score, top1, top5
- InputShape (int[]): Expected input dimensions (e.g., [3, 224, 224])
- OutputShape (int[]): Expected output dimensions
- NumParameters (long): Number of model parameters
- FileSizeBytes (long): Size of model file
- License (string): License type (e.g., "MIT", "Apache-2.0")
- PaperUrl (string): Link to research paper
- SourceCodeUrl (string): Link to source code
- Sha256Checksum (string): Hash for file integrity validation
- DownloadUrl (string): Primary download URL
- MirrorUrls (string[]): Fallback download URLs

### 2. ModelMetadataValidator
Create validation logic to ensure:
- Required fields are populated
- Version follows semantic versioning format
- URLs are valid
- SHA256 checksum is a valid 64-character hex string
- Performance metrics are in valid ranges (0-1 for accuracy)

### 3. Serialization Support
- Implement JSON serialization/deserialization
- Support reading from JSON files
- Support writing to JSON files

### 4. Unit Tests
Test cases for:
- Valid metadata creation
- Invalid metadata detection (missing fields, invalid versions, invalid checksums)
- Serialization/deserialization round-trip
- Edge cases (empty arrays, null values where allowed)

## Files to Create
- `src/ModelZoo/ModelMetadata.cs`
- `src/ModelZoo/ModelMetadataValidator.cs`
- `tests/ModelZooTests/ModelMetadataTests.cs`

## Dependencies
- System.Text.Json (for JSON serialization)
- System.ComponentModel.DataAnnotations (for validation attributes)

## Success Criteria
- Can create, validate, and serialize ModelMetadata objects
- Invalid metadata throws appropriate exceptions
- JSON round-trip preserves all data
- Test coverage > 90%
