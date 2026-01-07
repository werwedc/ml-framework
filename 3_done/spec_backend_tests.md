# Spec: Backend Abstraction Tests

## Overview
Implement tests for quantization backend abstraction layer.

## Requirements

### 1. IQuantizationBackend Interface Tests
- Test backend interface:
  - All backends implement interface correctly
  - `IsAvailable()` returns boolean
  - `GetName()` returns backend name
  - All required methods are implemented

### 2. BackendFactory Tests
- Test `BackendFactory`:
  - `CreateDefault()` returns best available backend
  - `Create(string name)` creates specific backend
  - `GetAvailableBackends()` returns all backends
  - `SetPreferredBackend()` sets preference
  - Invalid backend name throws exception

### 3. CPUBackend Tests
- Test `CPUBackend`:
  - `IsAvailable()` always returns true
  - `Quantize()` produces correct results
  - `Dequantize()` produces correct results
  - `MatMulInt8()` produces correct results
  - `Conv2DInt8()` produces correct results

### 4. x86Backend Tests (Conditional)
- Test `x86Backend` (if available):
  - `IsAvailable()` checks CPU features
  - `Quantize()` uses oneDNN acceleration
  - `MatMulInt8()` uses VNNI/AVX-512
  - Fallback to CPUBackend if unavailable

### 5. ARMBackend Tests (Conditional)
- Test `ARMBackend` (if available):
  - `IsAvailable()` checks ARM features
  - `Quantize()` uses NEON acceleration
  - `MatMulInt8()` uses dot product instructions
  - Fallback to CPUBackend if unavailable

### 6. GPUBackend Tests (Conditional)
- Test `GPUBackend` (if available):
  - `IsAvailable()` checks CUDA availability
  - `Quantize()` uses CUDA acceleration
  - `MatMulInt8()` uses Tensor Cores
  - Fallback to CPUBackend if unavailable

### 7. BackendCapabilityFlags Tests
- Test capability flags:
  - Flag values are correct
  - Flag combinations work correctly
  - Flag string representation

### 8. BackendCapabilities Tests
- Test backend capabilities:
  - Correctly reports supported operations
  - Correctly reports max tensor size
  - Correctly reports min batch size

### 9. Performance Comparison Tests
- Compare backend performance:
  - CPUBackend vs x86Backend (if available)
  - CPUBackend vs ARMBackend (if available)
  - CPUBackend vs GPUBackend (if available)
  - Measure speedup ratios

### 10. Backend Switching Tests
- Test dynamic backend switching:
  - Switch between backends during runtime
  - Verify backend switching is transparent
  - Verify results are consistent across backends

## File Structure
```
tests/
  MLFramework.Tests/
    Quantization/
      Backends/
        BackendInterfaceTests.cs
        BackendFactoryTests.cs
        CPUBackendTests.cs
        x86BackendTests.cs
        ARMBackendTests.cs
        GPUBackendTests.cs
        BackendCapabilitiesTests.cs
        PerformanceComparisonTests.cs
        BackendSwitchingTests.cs
```

## Implementation Notes
- Use conditional compilation for platform-specific tests
- Mock unavailable backends for CI/CD
- Use performance counters for timing
- Test with synthetic data (no dependencies on external models)

## Dependencies
- Core quantization components
- Backend abstraction components
- xUnit test framework
- System.Diagnostics for performance measurement
