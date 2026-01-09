# Mobile Runtime

Lightweight mobile runtime for ML model inference on iOS and Android.

## Features

- CPU backend with ARM NEON/SVE optimization
- Metal backend for iOS devices
- Vulkan backend for Android devices
- Lightweight tensor operations (no gradients)
- Memory pool for efficient memory management
- Compact model format with compression
- CLI tool for model conversion

## Installation

### NuGet

```bash
dotnet add package MobileRuntime
```

### Build from source

```bash
./build.sh    # Linux/macOS
./build.ps1   # Windows
```

## Usage

```csharp
using MobileRuntime;

// Create runtime
var runtime = new RuntimeMobileRuntime();

// Load model
var model = runtime.LoadModel("model.mob");

// Prepare input
var tensorFactory = new TensorFactory();
var input = tensorFactory.CreateTensor(data, new[] { 1, 28, 28 });

// Run inference
var outputs = model.Predict(new[] { input });
```

## Model Conversion

```bash
# Convert model to mobile format
MobileModelConverter convert -i model.onnx -o model.mob

# Convert with quantization
MobileModelConverter convert -i model.onnx -o model.mob --quantize int8
```

## Performance

- Model load time: < 100ms
- Inference latency: < 20ms
- Memory footprint: < 50MB
- Binary size: < 5MB

## Supported Platforms

- iOS 12.0+ (ARM64)
- Android 7.0+ (ARM64, ARMv7)
- .NET Standard 2.0+ (desktop)

## License

MIT
