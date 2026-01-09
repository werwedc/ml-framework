# Spec: Model Converter Tool

## Overview
Create a CLI tool to convert trained models to the mobile runtime binary format.

## Requirements
- Convert models from training framework to mobile format
- Optimize graph for inference-only execution
- Constant folding and dead code elimination
- Model size compression
- Quantization support (FP32 -> FP16/Int8)
- Validation against original model

## Classes to Implement

### 1. `ModelConverter` Class
```csharp
public class ModelConverter
{
    private readonly ConversionOptions _options;
    private readonly GraphOptimizer _optimizer;

    public ModelConverter(ConversionOptions options);

    public byte[] Convert(string modelPath);
    public byte[] Convert(byte[] modelData);
    public void Convert(string inputPath, string outputPath);

    private MobileModelFormat ProcessModel(ModelFormat inputFormat);
    private OperatorDescriptor[] OptimizeGraph(OperatorDescriptor[] operators);
    private ConstantTensor[] OptimizeConstants(ConstantTensor[] tensors);
    private void QuantizeWeights(ConstantTensor[] tensors, QuantizationOptions options);
    private void ValidateModel(MobileModelFormat model, ModelFormat originalModel);
}

public class ConversionOptions
{
    public QuantizationType Quantization { get; set; } = QuantizationType.None;
    public bool EnableGraphOptimization { get; set; } = true;
    public bool EnableConstantFolding { get; set; } = true;
    public bool EnableDeadCodeElimination { get; set; } = true;
    public bool EnableCompression { get; set; } = true;
    public CompressionLevel CompressionLevel { get; set; } = CompressionLevel.Default;
    public int TargetDeviceMemoryMB { get; set; } = 50;
}

public enum QuantizationType
{
    None,
    FP16,
    Int8,
    Int16
}

public enum CompressionLevel
{
    None,
    Fast,
    Default,
    Maximum
}
```

### 2. `GraphOptimizer` Class
```csharp
public class GraphOptimizer
{
    public GraphOptimizerResult Optimize(OperatorDescriptor[] operators, ConstantTensor[] constants);

    private OperatorDescriptor[] ConstantFolding(OperatorDescriptor[] operators, ConstantTensor[] constants);
    private OperatorDescriptor[] DeadCodeElimination(OperatorDescriptor[] operators);
    private OperatorDescriptor[] OperatorFusion(OperatorDescriptor[] operators);
    private OperatorDescriptor[] ReorderOperators(OperatorDescriptor[] operators);
}

public class GraphOptimizerResult
{
    public OperatorDescriptor[] OptimizedOperators { get; set; }
    public ConstantTensor[] OptimizedConstants { get; set; }
    public int OriginalOperatorCount { get; set; }
    public int OptimizedOperatorCount { get; set; }
    public int OriginalConstantBytes { get; set; }
    public int OptimizedConstantBytes { get; set; }
}
```

### 3. `ConstantFolder` Class
```csharp
public class ConstantFolder
{
    public ConstantTensor Fold(OperatorDescriptor op, ConstantTensor[] inputs);

    private ConstantTensor FoldAdd(ConstantTensor a, ConstantTensor b);
    private ConstantTensor FoldMultiply(ConstantTensor a, ConstantTensor b);
    private ConstantTensor FoldRelu(ConstantTensor input);
    private ConstantTensor FoldSigmoid(ConstantTensor input);
    private ConstantTensor FoldReshape(ConstantTensor input, int[] newShape);
}
```

### 4. `OperatorFuser` Class
```csharp
public class OperatorFuser
{
    public OperatorDescriptor[] Fuse(OperatorDescriptor[] operators);

    private bool CanFuse(OperatorDescriptor a, OperatorDescriptor b);
    private OperatorDescriptor FuseConv2DAndRelu(OperatorDescriptor conv, OperatorDescriptor relu);
    private OperatorDescriptor FuseConv2DAndBatchNorm(OperatorDescriptor conv, OperatorDescriptor bn);
    private OperatorDescriptor FuseFullyConnectedAndRelu(OperatorDescriptor fc, OperatorDescriptor relu);
}
```

### 5. `Quantizer` Class
```csharp
public class Quantizer
{
    public void Quantize(ConstantTensor[] tensors, QuantizationType type);

    private void QuantizeFP16(ConstantTensor tensor);
    private void QuantizeInt8(ConstantTensor tensor, float scale, int zeroPoint);
    private void QuantizeInt16(ConstantTensor tensor, float scale, int zeroPoint);

    private (float scale, int zeroPoint) CalculateScaleAndZeroPoint(float[] data);
    private float[] DequantizeInt8(byte[] quantized, float scale, int zeroPoint);
    private float[] DequantizeInt16(short[] quantized, float scale, int zeroPoint);
}
```

### 6. `ModelValidator` Class
```csharp
public class ModelValidator
{
    public ValidationResult Validate(MobileModelFormat mobileModel, ModelFormat originalModel);

    private bool ValidateShapeConsistency(MobileModelFormat model);
    private bool ValidateOperatorConnections(MobileModelFormat model);
    private bool ValidateDataTypes(MobileModelFormat model);
    private bool CompareOutputs(ITensor[] original, ITensor[] converted, float tolerance = 0.001f);
}
```

### 7. CLI Tool: `MobileModelConverter`

#### Program.cs
```csharp
using CommandLine;

class Program
{
    static int Main(string[] args)
    {
        return Parser.Default.ParseArguments<ConvertOptions, ValidateOptions, OptimizeOptions>(args)
            .MapResult(
                (ConvertOptions opts) => RunConvert(opts),
                (ValidateOptions opts) => RunValidate(opts),
                (OptimizeOptions opts) => RunOptimize(opts),
                errs => 1
            );
    }

    private static int RunConvert(ConvertOptions opts)
    {
        try
        {
            var converter = new ModelConverter(opts.ToConversionOptions());
            converter.Convert(opts.InputModel, opts.OutputModel);

            Console.WriteLine($"Model converted successfully: {opts.OutputModel}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunValidate(ValidateOptions opts)
    {
        try
        {
            var validator = new ModelValidator();
            var result = validator.Validate(opts.InputModel, opts.ReferenceModel);

            if (result.IsValid)
            {
                Console.WriteLine("Model validation passed");
                return 0;
            }
            else
            {
                Console.Error.WriteLine("Model validation failed:");
                foreach (var error in result.Errors)
                {
                    Console.Error.WriteLine($"  - {error}");
                }
                return 1;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunOptimize(OptimizeOptions opts)
    {
        try
        {
            var optimizer = new GraphOptimizer();
            // Implementation...
            Console.WriteLine("Model optimization completed");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }
}
```

#### Options.cs
```csharp
class ConvertOptions
{
    [Option('i', "input", Required = true, HelpText = "Input model path")]
    public string InputModel { get; set; }

    [Option('o', "output", Required = true, HelpText = "Output model path")]
    public string OutputModel { get; set; }

    [Option('q', "quantize", Required = false, HelpText = "Quantization type (none, fp16, int8, int16)", Default = "none")]
    public string Quantization { get; set; }

    [Option('c', "compress", Required = false, HelpText = "Enable compression", Default = true)]
    public bool EnableCompression { get; set; }

    [Option('m', "memory", Required = false, HelpText = "Target device memory in MB", Default = 50)]
    public int TargetMemoryMB { get; set; }

    [Option('v', "verbose", Required = false, HelpText = "Verbose output", Default = false)]
    public bool Verbose { get; set; }

    public ConversionOptions ToConversionOptions()
    {
        return new ConversionOptions
        {
            Quantization = Enum.Parse<QuantizationType>(Quantization, true),
            EnableCompression = EnableCompression,
            TargetDeviceMemoryMB = TargetMemoryMB
        };
    }
}

class ValidateOptions
{
    [Option('i', "input", Required = true, HelpText = "Converted model path")]
    public string InputModel { get; set; }

    [Option('r', "reference", Required = true, HelpText = "Reference model path")]
    public string ReferenceModel { get; set; }

    [Option('t', "tolerance", Required = false, HelpText = "Output tolerance", Default = 0.001)]
    public float Tolerance { get; set; }
}

class OptimizeOptions
{
    [Option('i', "input", Required = true, HelpText = "Input model path")]
    public string InputModel { get; set; }

    [Option('o', "output", Required = true, HelpText = "Output model path")]
    public string OutputModel { get; set; }

    [Option('f', "fold", Required = false, HelpText = "Enable constant folding", Default = true)]
    public bool EnableConstantFolding { get; set; }

    [Option('d', "dead-code", Required = false, HelpText = "Enable dead code elimination", Default = true)]
    public bool EnableDeadCodeElimination { get; set; }
}
```

## Usage Examples

### Convert model with default settings
```bash
MobileModelConverter convert -i model.onnx -o model.mob
```

### Convert with FP16 quantization
```bash
MobileModelConverter convert -i model.onnx -o model.mob --quantize fp16
```

### Convert with Int8 quantization and compression
```bash
MobileModelConverter convert -i model.onnx -o model.mob --quantize int8 --compress
```

### Validate converted model
```bash
MobileModelConverter validate -i model.mob -r reference.onnx
```

### Optimize model only
```bash
MobileModelConverter optimize -i model.onnx -o model_optimized.onnx
```

## Implementation Notes

### Conversion Pipeline
1. Parse input model (ONNX, Protobuf, etc.)
2. Validate model structure
3. Apply graph optimizations
4. Quantize weights (if enabled)
5. Fold constants (if enabled)
6. Compress model (if enabled)
7. Serialize to mobile format
8. Validate outputs match original

### Constant Folding
- Execute operators with constant inputs
- Replace subgraph with constant tensor
- Reduce model size
- Improve inference speed

### Operator Fusion
- Fuse Conv2D + BatchNorm
- Fuse Conv2D + Relu
- Fuse FullyConnected + Relu
- Reduce kernel launches

### Quantization
- FP16: 16-bit floating point (2x compression)
- Int8: 8-bit integer (4x compression) with calibration
- Int16: 16-bit integer (2x compression) with calibration
- Scale and zero-point computation

### Compression
- GZip compression for constant tensors
- Delta encoding for weights
- Huffman encoding for activation patterns

## File Structure
```
tools/MobileModelConverter/
├── Program.cs
├── Options.cs
├── ModelConverter.cs
├── GraphOptimizer.cs
├── ConstantFolder.cs
├── OperatorFuser.cs
├── Quantizer.cs
├── ModelValidator.cs
└── Models/
    ├── ConversionOptions.cs
    ├── GraphOptimizerResult.cs
    └── ValidationResult.cs
```

## Success Criteria
- Converts models successfully
- Optimized models are smaller
- Output accuracy matches within tolerance
- CLI tool works as expected
- Performance meets targets

## Dependencies
- spec_mobile_model_format (MobileModelFormat)
- CommandLineParser (NuGet)
- System.Compression (for GZip)

## Testing Requirements
- Unit tests for conversion pipeline
- Unit tests for graph optimizations
- Unit tests for quantization
- Integration tests with real models
- Accuracy validation tests

## Performance Targets
- Conversion time: < 5s for typical models
- Model size reduction: > 50%
- Output accuracy: > 99.9% match with original
