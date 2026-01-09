# Spec: Native Model Serialization Format

## Overview
Implement a binary serialization format for storing and loading model weights optimized for the framework.

## Requirements

### 1. File Format Specification
Define a binary format with the following structure:
- Header (magic bytes, version info)
- Metadata section (model architecture, layer names, shapes)
- Weights section (tensor data in binary)
- Footer (checksum for integrity)

Format layout:
```
[Header: 16 bytes]
  - Magic: "MLFW" (4 bytes)
  - Version: uint16 (2 bytes)
  - Flags: uint16 (2 bytes - compression, precision, etc.)
  - HeaderChecksum: uint32 (4 bytes)
  - Reserved: 4 bytes

[Metadata Section: variable length]
  - MetadataLength: uint32 (4 bytes)
  - MetadataData: bytes (JSON metadata including architecture, layer names, dtypes, shapes)

[Weights Section: variable length]
  - NumLayers: uint32 (4 bytes)
  - For each layer:
    - LayerNameLength: uint16
    - LayerName: bytes
    - TensorData: bytes (row-major float32/int32/etc.)

[Footer: 8 bytes]
  - FileChecksum: uint64 (SHA256 truncated or CRC64)
```

### 2. ModelSerializer Class
Serialization operations:
- `Serialize(Model model, Stream output, SerializerOptions options = null)`: Serialize model to stream
- `SerializeToFile(Model model, string path, SerializerOptions options = null)`: Serialize to file
- `GetSerializedSize(Model model)`: Calculate size without serializing

### 3. ModelDeserializer Class
Deserialization operations:
- `Deserialize(Stream input, Device device = null)`: Deserialize from stream
- `DeserializeFromFile(string path, Device device = null)`: Deserialize from file
- `VerifyChecksum(string path)`: Verify file integrity

### 4. SerializerOptions
Configuration options:
- Precision (FP32, FP16, BF16)
- Compression (None, GZip, Zstd)
- IncludeOptimizerState (bool)
- MetadataOnly (bool - for inspection)

### 5. Tensor Serialization
Handle tensor data:
- Support float32, float16, int32, int64, bool tensors
- Preserve tensor shape information
- Row-major storage order
- Optional compression per tensor

### 6. Unit Tests
Test cases for:
- Serialize and deserialize simple model
- Serialize and deserialize ResNet-50
- Checksum verification
- Different precision options
- Compression support
- Metadata inspection without loading weights
- Error cases (corrupted file, invalid magic bytes, version mismatch)
- Large model serialization (> 1GB)

## Files to Create
- `src/ModelZoo/Serialization/ModelSerializer.cs`
- `src/ModelZoo/Serialization/ModelDeserializer.cs`
- `src/ModelZoo/Serialization/SerializerOptions.cs`
- `src/ModelZoo/Serialization/FileFormatSpec.cs`
- `tests/ModelZooTests/SerializationTests.cs`

## Dependencies
- System.IO (for file/stream operations)
- System.IO.Compression (for compression)
- System.IO.Hashing (for checksums)
- Existing Model/Tensor infrastructure

## Success Criteria
- Can serialize and deserialize models with perfect fidelity
- File format is backward compatible (can read old versions)
- Checksums detect corruption
- Compression reduces file size by 30-50%
- Test coverage > 90%
