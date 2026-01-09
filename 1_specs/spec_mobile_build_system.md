# Spec: Build System and NuGet Package

## Overview
Create build system, NuGet packages, and cross-platform configuration for the mobile runtime.

## Requirements
- Separate NuGet package for mobile runtime
- Conditional compilation for mobile targets
- Platform-specific native library bundling
- AOT compilation support
- Automated build and release pipeline

## Files to Create

### 1. Project File: `MobileRuntime.csproj`

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFrameworks>netstandard2.0;net6.0-android;net6.0-ios</TargetFrameworks>
    <LangVersion>latest</LangVersion>
    <Nullable>enable</Nullable>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
    <NoWarn>CS1591</NoWarn>
    <RootNamespace>MobileRuntime</RootNamespace>
    <AssemblyName>MobileRuntime</AssemblyName>
    <Version>1.0.0</Version>
    <Authors>ML Framework Team</Authors>
    <Company>ML Framework</Company>
    <Description>Lightweight mobile runtime for ML model inference on iOS and Android</Description>
    <PackageTags>machine-learning;mobile;inference;ios;android;metal;vulkan</PackageTags>
    <PackageProjectUrl>https://github.com/your-org/ml-framework</PackageProjectUrl>
    <RepositoryUrl>https://github.com/your-org/ml-framework</RepositoryUrl>
    <RepositoryType>git</RepositoryUrl>
    <PackageLicenseExpression>MIT</PackageLicenseExpression>
    <PackageRequireLicenseAcceptance>false</PackageRequireLicenseAcceptance>
    <PackageIcon>icon.png</PackageIcon>
    <PackageReadmeFile>README.md</PackageReadmeFile>
    <PackageReleaseNotes>
      Initial release of Mobile Runtime
      - CPU backend with ARM NEON/SVE optimization
      - Metal backend for iOS
      - Vulkan backend for Android
      - Lightweight tensor operations
      - Memory pool for efficient memory management
    </PackageReleaseNotes>
  </PropertyGroup>

  <PropertyGroup Condition="'$(TargetFramework)' == 'netstandard2.0'">
    <DefineConstants>NETSTANDARD</DefineConstants>
  </PropertyGroup>

  <PropertyGroup Condition="'$(TargetFramework)' == 'net6.0-android'">
    <DefineConstants>ANDROID;NET6_0_OR_GREATER</DefineConstants>
    <SupportedOSPlatformVersion>21.0</SupportedOSPlatformVersion>
    <AndroidEnablePackageValidation>false</AndroidEnablePackageValidation>
  </PropertyGroup>

  <PropertyGroup Condition="'$(TargetFramework)' == 'net6.0-ios'">
    <DefineConstants>IOS;NET6_0_OR_GREATER</DefineConstants>
    <SupportedOSPlatformVersion>12.0</SupportedOSPlatformVersion>
    <MtouchEnableSGenConc>true</MtouchEnableSGenConc>
    <MtouchHttpClientHandler>NSUrlSessionHandler</MtouchHttpClientHandler>
  </PropertyGroup>

  <ItemGroup>
    <None Include="../../images/icon.png" Pack="true" PackagePath="\" />
    <None Include="../../README.md" Pack="true" PackagePath="\" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="System.Memory" Version="4.5.5" />
    <PackageReference Include="System.Runtime.CompilerServices.Unsafe" Version="6.0.0" />
  </ItemGroup>

  <!-- Android-specific references -->
  <ItemGroup Condition="'$(TargetFramework)' == 'net6.0-android'">
    <PackageReference Include="Xamarin.AndroidX.AppCompat" Version="1.3.1" />
  </ItemGroup>

  <!-- iOS-specific references -->
  <ItemGroup Condition="'$(TargetFramework)' == 'net6.0-ios'">
    <PackageReference Include="Xamarin.iOS" Version="15.0.0" />
  </ItemGroup>

  <!-- Native libraries -->
  <ItemGroup Condition="'$(TargetFramework)' == 'net6.0-android'">
    <AndroidNativeLibrary Include="libs/android/arm64-v8a/libvulkan.so" />
    <AndroidNativeLibrary Include="libs/android/armeabi-v7a/libvulkan.so" />
    <AndroidNativeLibrary Include="libs/android/x86_64/libvulkan.so" />
  </ItemGroup>

  <ItemGroup Condition="'$(TargetFramework)' == 'net6.0-ios'">
    <NativeReference Include="libs/ios/libMobileRuntime.framework">
      <SmartLink>true</SmartLink>
      <ForceLoad>true</ForceLoad>
      <IsCxx>true</IsCxx>
    </NativeReference>
  </ItemGroup>

  <!-- Source files -->
  <ItemGroup>
    <Compile Include="Interfaces/IMobileRuntime.cs" />
    <Compile Include="Interfaces/IModel.cs" />
    <Compile Include="Interfaces/ITensor.cs" />
    <Compile Include="Interfaces/IBackend.cs" />
    <!-- Add other source files -->
  </ItemGroup>

</Project>
```

### 2. Project File: `MobileRuntime.Tests.csproj`

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFrameworks>net6.0;net6.0-android;net6.0-ios</TargetFrameworks>
    <LangVersion>latest</LangVersion>
    <Nullable>enable</Nullable>
    <IsPackable>false</IsPackable>
    <IsTestProject>true</IsTestProject>
    <RootNamespace>MobileRuntime.Tests</RootNamespace>
    <AssemblyName>MobileRuntime.Tests</AssemblyName>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.0.0" />
    <PackageReference Include="MSTest.TestAdapter" Version="2.2.8" />
    <PackageReference Include="MSTest.TestFramework" Version="2.2.8" />
    <PackageReference Include="coverlet.collector" Version="3.1.2" />
    <PackageReference Include="FluentAssertions" Version="6.2.0" />
  </ItemGroup>

  <ItemGroup>
    <ProjectReference Include="..\MobileRuntime\MobileRuntime.csproj" />
  </ItemGroup>

  <ItemGroup>
    <Compile Include="TensorOperationsTests.cs" />
    <Compile Include="MemoryPoolTests.cs" />
    <Compile Include="ModelFormatTests.cs" />
    <Compile Include="CpuBackendTests.cs" />
    <Compile Include="MobileModelTests.cs" />
    <Compile Include="IntegrationTests.cs" />
    <Compile Include="PerformanceTests.cs" />
    <Compile Include="MemoryLeakTests.cs" />
  </ItemGroup>

</Project>
```

### 3. Project File: `MobileModelConverter.csproj`

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <LangVersion>latest</LangVersion>
    <Nullable>enable</Nullable>
    <RootNamespace>MobileModelConverter</RootNamespace>
    <AssemblyName>MobileModelConverter</AssemblyName>
    <Version>1.0.0</Version>
    <Authors>ML Framework Team</Authors>
    <Description>CLI tool to convert ML models to mobile runtime format</Description>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="CommandLineParser" Version="2.9.0" />
    <PackageReference Include="System.Text.Json" Version="6.0.0" />
  </ItemGroup>

  <ItemGroup>
    <ProjectReference Include="..\MobileRuntime\MobileRuntime.csproj" />
  </ItemGroup>

  <ItemGroup>
    <Compile Include="Program.cs" />
    <Compile Include="Options.cs" />
    <Compile Include="ModelConverter.cs" />
    <Compile Include="GraphOptimizer.cs" />
    <Compile Include="ConstantFolder.cs" />
    <Compile Include="OperatorFuser.cs" />
    <Compile Include="Quantizer.cs" />
    <Compile Include="ModelValidator.cs" />
  </ItemGroup>

</Project>
```

### 4. Build Script: `build.sh`

```bash
#!/bin/bash

set -e

echo "Building Mobile Runtime..."

# Build all targets
dotnet build src/MobileRuntime/MobileRuntime.csproj --configuration Release

# Build tests
dotnet build tests/MobileRuntime.Tests/MobileRuntime.Tests.csproj --configuration Release

# Build converter tool
dotnet build tools/MobileModelConverter/MobileModelConverter.csproj --configuration Release

# Run tests
echo "Running tests..."
dotnet test tests/MobileRuntime.Tests/MobileRuntime.Tests.csproj --configuration Release --no-build

# Create NuGet package
echo "Creating NuGet package..."
dotnet pack src/MobileRuntime/MobileRuntime.csproj --configuration Release --no-build

echo "Build completed successfully!"
```

### 5. Build Script: `build.ps1`

```powershell
$ErrorActionPreference = "Stop"

Write-Host "Building Mobile Runtime..." -ForegroundColor Green

# Build all targets
dotnet build src/MobileRuntime/MobileRuntime.csproj --configuration Release

# Build tests
dotnet build tests/MobileRuntime.Tests/MobileRuntime.Tests.csproj --configuration Release

# Build converter tool
dotnet build tools/MobileModelConverter/MobileModelConverter.csproj --configuration Release

# Run tests
Write-Host "Running tests..." -ForegroundColor Yellow
dotnet test tests/MobileRuntime.Tests/MobileRuntime.Tests.csproj --configuration Release --no-build

# Create NuGet package
Write-Host "Creating NuGet package..." -ForegroundColor Yellow
dotnet pack src/MobileRuntime/MobileRuntime.csproj --configuration Release --no-build

Write-Host "Build completed successfully!" -ForegroundColor Green
```

### 6. GitHub Actions Workflow: `.github/workflows/build.yml`

```yaml
name: Build and Test

on:
  push:
    branches: [ main, ai ]
  pull_request:
    branches: [ main, ai ]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        dotnet: [6.0.x]

    steps:
    - uses: actions/checkout@v3

    - name: Setup .NET
      uses: actions/setup-dotnet@v3
      with:
        dotnet-version: ${{ matrix.dotnet }}

    - name: Restore dependencies
      run: dotnet restore

    - name: Build
      run: dotnet build --configuration Release --no-restore

    - name: Test
      run: dotnet test --configuration Release --no-build --verbosity normal

    - name: Pack
      run: dotnet pack --configuration Release --no-build

    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: nupkg
        path: src/MobileRuntime/bin/Release/*.nupkg

  android-build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup .NET
      uses: actions/setup-dotnet@v3
      with:
        dotnet-version: 6.0.x

    - name: Build Android
      run: dotnet build src/MobileRuntime/MobileRuntime.csproj --framework net6.0-android --configuration Release

  ios-build:
    runs-on: macos-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup .NET
      uses: actions/setup-dotnet@v3
      with:
        dotnet-version: 6.0.x

    - name: Build iOS
      run: dotnet build src/MobileRuntime/MobileRuntime.csproj --framework net6.0-ios --configuration Release
```

### 7. Directory Structure

```
MobileRuntime/
├── src/
│   └── MobileRuntime/
│       ├── Interfaces/
│       ├── Models/
│       ├── Tensors/
│       ├── Memory/
│       ├── Serialization/
│       ├── Backends/
│       │   ├── Cpu/
│       │   ├── Metal/
│       │   ├── Vulkan/
│       │   └── Adapters/
│       ├── Benchmarking/
│       └── MobileRuntime.csproj
├── tests/
│   └── MobileRuntime.Tests/
│       ├── TensorOperationsTests.cs
│       ├── MemoryPoolTests.cs
│       ├── ModelFormatTests.cs
│       ├── CpuBackendTests.cs
│       ├── MobileModelTests.cs
│       ├── IntegrationTests.cs
│       ├── PerformanceTests.cs
│       ├── MemoryLeakTests.cs
│       └── MobileRuntime.Tests.csproj
├── tools/
│   └── MobileModelConverter/
│       ├── Program.cs
│       ├── Options.cs
│       ├── ModelConverter.cs
│       ├── GraphOptimizer.cs
│       ├── ConstantFolder.cs
│       ├── OperatorFuser.cs
│       ├── Quantizer.cs
│       ├── ModelValidator.cs
│       └── MobileModelConverter.csproj
├── libs/
│   ├── android/
│   │   ├── arm64-v8a/
│   │   │   └── libvulkan.so
│   │   ├── armeabi-v7a/
│   │   │   └── libvulkan.so
│   │   └── x86_64/
│   │       └── libvulkan.so
│   └── ios/
│       └── libMobileRuntime.framework/
├── build.sh
├── build.ps1
├── README.md
└── .github/
    └── workflows/
        └── build.yml
```

### 8. README.md

```markdown
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
```

## Implementation Notes

### Conditional Compilation
- Use `#if ANDROID` for Android-specific code
- Use `#if IOS` for iOS-specific code
- Use `#if NETSTANDARD` for shared code

### Native Libraries
- Bundle pre-compiled native libraries
- Use P/Invoke for native API calls
- Handle library loading gracefully (fallback to CPU)

### AOT Compilation
- Use `[NativeDependency]` for native references
- Mark methods with `[DllImport]` attributes
- Test AOT builds for iOS

## Success Criteria

- Build succeeds on all platforms
- NuGet package is created
- Tests pass on all platforms
- Native libraries are bundled correctly
- CI/CD pipeline works

## Dependencies

- .NET 6.0 SDK
- Xcode (for iOS builds)
- Android SDK (for Android builds)
- Vulkan SDK (for Android builds)

## Platform Requirements

- Windows: .NET 6.0 SDK
- Linux: .NET 6.0 SDK, Vulkan SDK
- macOS: .NET 6.0 SDK, Xcode
- Android: .NET 6.0 SDK, Android SDK, NDK
- iOS: .NET 6.0 SDK, Xcode, physical device
