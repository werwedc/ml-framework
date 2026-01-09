# Spec: Model Format and Serialization

## Overview
Define a compact binary format for mobile runtime models optimized for fast loading and minimal memory footprint.

## Requirements
- Compact binary format (< 50% of original size)
- Fast parsing (< 100ms for MNIST/CIFAR models)
- Supports quantization (Int8/Int16/FP16)
- Includes metadata for inputs/outputs
- Versioned format for backward compatibility
- Platform-independent byte order (little-endian)

## File Format Specification

### Binary Layout (Big Picture)
```
[Header]
[Model Metadata]
[Input/Output Specs]
[Constant Tensors]
[Operator Graph]
[Operator Parameters]
[Checksum]
```

### Header (32 bytes)
```
Offset | Size | Description
-------|------|------------
0      | 4    | Magic number (0x4D4F4249 = "MOBI")
4      | 2    | Format version (uint16)
6      | 2    | Reserved (uint16)
8      | 4    | Header checksum (uint32)
12     | 4    | Flags (uint32, bitmask)
16     | 4    | Total file size (uint32)
20     | 4    | Model metadata offset (uint32)
24     | 4    | Tensor data offset (uint32)
28     | 4    | Operator graph offset (uint32)
```

### Flags
```
Bit 0: Model is quantized
Bit 1: Uses FP16 weights
Bit 2: Graph is optimized (operators fused)
Bit 3-31: Reserved
```

### Model Metadata
```
Offset | Size | Description
-------|------|------------
0      | 4    | Model name length (uint32)
4      | N    | Model name (UTF-8)
4+N    | 4    | Framework version (uint32)
8+N    | 4    | Creation timestamp (uint64, Unix time)
16+N   | 4    | Input count (uint32)
20+N   | 4    | Output count (uint32)
```

### Input/Output Spec (per input/output)
```
Offset | Size | Description
-------|------|------------
0      | 4    | Name length (uint32)
4      | N    | Name (UTF-8)
4+N    | 2    | Rank (uint16)
6+N    | 2    | Data type (uint16, see DataType enum)
8+N    | 8*Rank| Shape (uint64 array)
```

### Constant Tensor Header
```
Offset | Size | Description
-------|------|------------
0      | 4    | Tensor ID (uint32)
4      | 2    | Data type (uint16)
6      | 2    | Rank (uint16)
8      | 8*Rank| Shape (uint64 array)
8+8*Rank| 8   | Data size in bytes (uint64)
16+8*Rank| N   | Tensor data (binary)
```

### Operator Graph
```
Offset | Size | Description
-------|------|------------
0      | 4    | Operator count (uint32)
4      | N    | Operator descriptors (variable length)
```

### Operator Descriptor
```
Offset | Size | Description
-------|------|------------
0      | 2    | Operator type (uint16, see OperatorType enum)
2      | 2    | Input count (uint16)
4      | 2    | Output count (uint16)
6      | 2    | Parameter count (uint16)
8      | 4*InputCount | Input tensor IDs (uint32 array)
8+4*InputCount | 4*OutputCount | Output tensor IDs (uint32 array)
8+4*InputCount+4*OutputCount | M | Operator parameters (variable length)
```

### Checksum (footer)
```
Offset | Size | Description
-------|------|------------
FileEnd-4 | 4 | CRC32 checksum of entire file
```

## Classes to Implement

### 1. `MobileModelFormat` Class (Reader)
```csharp
public sealed class MobileModelFormat : IDisposable
{
    private readonly BinaryReader _reader;
    private readonly Stream _stream;
    private ModelHeader _header;
    private ModelMetadata _metadata;
    private List<InputOutputSpec> _inputs;
    private List<InputOutputSpec> _outputs;
    private Dictionary<uint, ConstantTensor> _constantTensors;
    private List<OperatorDescriptor> _operators;

    public MobileModelFormat(string filePath);
    public MobileModelFormat(Stream stream);
    public MobileModelFormat(byte[] data);

    public void Read();

    public ModelHeader Header => _header;
    public ModelMetadata Metadata => _metadata;
    public InputOutputSpec[] Inputs => _inputs.ToArray();
    public InputOutputSpec[] Outputs => _outputs.ToArray();
    public ConstantTensor[] ConstantTensors => _constantTensors.Values.ToArray();
    public OperatorDescriptor[] Operators => _operators.ToArray();

    public void Dispose();

    private void ValidateHeader();
    private void ReadMetadata();
    private void ReadInputOutputSpecs();
    private void ReadConstantTensors();
    private void ReadOperatorGraph();
    private void ValidateChecksum();
}

public class ModelHeader
{
    public uint MagicNumber { get; set; }
    public ushort Version { get; set; }
    public uint HeaderChecksum { get; set; }
    public uint Flags { get; set; }
    public uint TotalFileSize { get; set; }
    public uint ModelMetadataOffset { get; set; }
    public uint TensorDataOffset { get; set; }
    public uint OperatorGraphOffset { get; set; }
}

public class ModelMetadata
{
    public string Name { get; set; }
    public uint FrameworkVersion { get; set; }
    public ulong CreationTimestamp { get; set; }
    public uint InputCount { get; set; }
    public uint OutputCount { get; set; }
}

public class InputOutputSpec
{
    public string Name { get; set; }
    public ushort Rank { get; set; }
    public DataType DataType { get; set; }
    public ulong[] Shape { get; set; }
}

public class ConstantTensor
{
    public uint Id { get; set; }
    public DataType DataType { get; set; }
    public ushort Rank { get; set; }
    public ulong[] Shape { get; set; }
    public ulong DataSize { get; set; }
    public byte[] Data { get; set; }

    public Tensor ToTensor();
}

public class OperatorDescriptor
{
    public OperatorType Type { get; set; }
    public uint[] InputTensorIds { get; set; }
    public uint[] OutputTensorIds { get; set; }
    public Dictionary<string, object> Parameters { get; set; }
}

public enum OperatorType : ushort
{
    Conv2D,
    DepthwiseConv2D,
    FullyConnected,
    MaxPool2D,
    AvgPool2D,
    BatchNorm,
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Add,
    Subtract,
    Multiply,
    Divide,
    Concat,
    Reshape,
    Flatten,
    Transpose,
    MatMul,
    // Add more as needed
}
```

### 2. `MobileModelWriter` Class (Writer)
```csharp
public sealed class MobileModelWriter
{
    private readonly BinaryWriter _writer;
    private readonly MemoryStream _stream;

    public MobileModelWriter();
    public MobileModelWriter(Stream stream);

    public byte[] WriteModel(ModelWriterConfig config);

    private void WriteHeader(ModelWriterConfig config);
    private void WriteMetadata(ModelWriterConfig config);
    private void WriteInputOutputSpecs(InputOutputSpec[] specs);
    private void WriteConstantTensors(ConstantTensor[] tensors);
    private void WriteOperatorGraph(OperatorDescriptor[] operators);
    private uint CalculateChecksum();
}

public class ModelWriterConfig
{
    public string ModelName { get; set; }
    public uint FrameworkVersion { get; set; }
    public InputOutputSpec[] Inputs { get; set; }
    public InputOutputSpec[] Outputs { get; set; }
    public ConstantTensor[] ConstantTensors { get; set; }
    public OperatorDescriptor[] Operators { get; set; }
    public uint Flags { get; set; }
}
```

### 3. `ModelSerializer` Utility Class
```csharp
public static class ModelSerializer
{
    // Load from file or bytes
    public static MobileModelFormat Load(string filePath);
    public static MobileModelFormat Load(byte[] data);
    public static MobileModelFormat Load(Stream stream);

    // Save to file or bytes
    public static byte[] Save(ModelWriterConfig config);
    public static void Save(string filePath, ModelWriterConfig config);

    // Validation
    public static bool Validate(string filePath);
    public static bool Validate(byte[] data);

    // Utility methods
    public static InputOutputSpec CreateSpec(string name, DataType dataType, params int[] shape);
    public static ConstantTensor CreateTensor(uint id, Tensor tensor);
    public static ConstantTensor CreateTensor(uint id, int[] shape, DataType dataType, byte[] data);
}
```

## Implementation Notes

### Byte Order
- Use `BitConverter.IsLittleEndian` to detect endianness
- Always write in little-endian format
- Read with little-endian conversion on big-endian systems

### Checksum
- Use CRC32 for header validation
- Use Adler32 for full file checksum (faster)
- Validate on read, throw exception if invalid

### Compression
- GZip compress constant tensor data
- Store compression flag in tensor header
- Decompress on read

## File Structure
```
src/MobileRuntime/Serialization/
├── MobileModelFormat.cs
├── MobileModelWriter.cs
├── ModelSerializer.cs
└── Models/
    ├── ModelHeader.cs
    ├── ModelMetadata.cs
    ├── InputOutputSpec.cs
    ├── ConstantTensor.cs
    └── OperatorDescriptor.cs
```

## Success Criteria
- Format specification complete
- Reader parses all sections correctly
- Writer creates valid files
- Checksum validation works
- File size < 50% of original model
- Load time < 100ms for MNIST models

## Dependencies
- spec_mobile_runtime_core (DataType enum)
- spec_mobile_tensor_ops (Tensor class)

## Testing Requirements
- Write/read round-trip tests
- Checksum validation tests
- Corrupted file handling tests
- Version compatibility tests
- Performance benchmarks (load time, file size)
- Cross-platform tests (byte order)

## Performance Targets
- Parse header: < 1ms
- Load MNIST model: < 50ms
- Load CIFAR model: < 100ms
- File size: < 2MB for MNIST, < 5MB for CIFAR
