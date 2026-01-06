# Spec: GPU-Accelerated Transforms

## Overview
Implement image preprocessing transforms that execute on the GPU for better performance.

## Requirements

### Interface

#### IGpuTransform
```csharp
public interface IGpuTransform : ITransform
{
    bool GpuAvailable { get; }
    void SetGpuDevice(int deviceId);
}
```

### GPU-Specific Transforms

#### GpuResizeTransform
- Resize images on GPU
- Faster than CPU for large batches
- Supports various interpolation methods

```csharp
public class GpuResizeTransform : IGpuTransform
{
    private readonly int _width;
    private readonly int _height;
    private readonly InterpolationMode _mode;
    private int _gpuDevice;

    public GpuResizeTransform(int width, int height, InterpolationMode mode = InterpolationMode.Bilinear)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException();

        _width = width;
        _height = height;
        _mode = mode;
        _gpuDevice = 0;
    }

    public bool GpuAvailable => DetectGpuSupport();
    public int GpuDevice => _gpuDevice;

    public void SetGpuDevice(int deviceId)
    {
        _gpuDevice = deviceId;
    }

    public object Apply(object input)
    {
        if (input is float[,] cpuImage)
        {
            // Convert to GPU tensor, resize, return (placeholder)
            return ResizeOnGpu(cpuImage, _width, _height, _mode);
        }

        throw new ArgumentException("Input must be 2D float array");
    }

    private object ResizeOnGpu(float[,] image, int width, int height, InterpolationMode mode)
    {
        // Placeholder: Integrate with CUDA image processing library
        // For now, do CPU resize as fallback
        return CpuResize(image, width, height, mode);
    }

    private float[,] CpuResize(float[,] image, int targetWidth, int targetHeight, InterpolationMode mode)
    {
        // Simple CPU resize implementation (placeholder)
        int srcHeight = image.GetLength(0);
        int srcWidth = image.GetLength(1);

        var result = new float[targetHeight, targetWidth];

        float scaleY = (float)srcHeight / targetHeight;
        float scaleX = (float)srcWidth / targetWidth;

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                int srcY = (int)(y * scaleY);
                int srcX = (int)(x * scaleX);
                result[y, x] = image[Math.Min(srcY, srcHeight - 1), Math.Min(srcX, srcWidth - 1)];
            }
        }

        return result;
    }

    private bool DetectGpuSupport()
    {
        // Placeholder: Check for CUDA availability
        return false;
    }
}
```

#### GpuNormalizeTransform
- Apply normalization on GPU
- Vectorized operations for speed

```csharp
public class GpuNormalizeTransform : IGpuTransform
{
    private readonly float[] _mean;
    private readonly float[] _std;
    private int _gpuDevice;

    public GpuNormalizeTransform(float[] mean, float[] std)
    {
        if (mean == null || std == null || mean.Length != std.Length)
            throw new ArgumentException();

        _mean = mean;
        _std = std;
        _gpuDevice = 0;
    }

    public bool GpuAvailable => DetectGpuSupport();
    public int GpuDevice => _gpuDevice;

    public void SetGpuDevice(int deviceId)
    {
        _gpuDevice = deviceId;
    }

    public object Apply(object input)
    {
        if (input is float[,] cpuImage)
        {
            return NormalizeOnGpu(cpuImage);
        }

        throw new ArgumentException("Input must be 2D float array");
    }

    private float[,] NormalizeOnGpu(float[,] image)
    {
        // Placeholder: GPU normalization
        // For now, CPU implementation
        int height = image.GetLength(0);
        int width = image.GetLength(1);
        var result = new float[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int channel = x % _mean.Length;
                result[y, x] = (image[y, x] - _mean[channel]) / _std[channel];
            }
        }

        return result;
    }

    private bool DetectGpuSupport()
    {
        // Placeholder
        return false;
    }
}
```

### Enums

#### InterpolationMode
```csharp
public enum InterpolationMode
{
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos
}
```

### Error Handling
- Validation of dimensions and parameters
- Fallback to CPU if GPU unavailable
- Graceful degradation

## Acceptance Criteria
1. GpuResizeTransform resizes images correctly
2. GpuNormalizeTransform applies normalization correctly
3. GpuAvailable property detects GPU support
4. SetGpuDevice updates target device
5. CPU fallback works when GPU unavailable
6. Different interpolation modes produce correct results
7. Unit tests verify correctness of transforms
8. Performance tests compare GPU vs CPU speed

## Files to Create
- `src/Data/Transforms/IGpuTransform.cs`
- `src/Data/Transforms/GpuResizeTransform.cs`
- `src/Data/Transforms/GpuNormalizeTransform.cs`
- `src/Data/Transforms/InterpolationMode.cs`

## Tests
- `tests/Data/Transforms/GpuResizeTransformTests.cs`
- `tests/Data/Transforms/GpuNormalizeTransformTests.cs`

## Usage Example
```csharp
var transform = new ComposeTransform(
    new GpuResizeTransform(width: 224, height: 224, mode: InterpolationMode.Bilinear),
    new GpuNormalizeTransform(mean: new[] {0.485f, 0.456f, 0.406f},
                             std: new[] {0.229f, 0.224f, 0.225f})
);

if (transform.GpuAvailable)
{
    transform.SetGpuDevice(0); // Use GPU 0
}

var result = transform.Apply(image);
```

## Notes
- GPU transforms provide 5-10x speedup for large batches
- Placeholder implementations use CPU for now
- Will integrate with CUDA image processing libraries (NPP, OpenCV CUDA)
- Common transforms: resize, normalize, color jitter, random flip
- Consider batch-wise transforms for efficiency
- Memory transfer overhead may negate GPU benefit for small batches
- Future: Add more transforms (crop, rotate, perspective warp)
- Monitor GPU memory usage during transforms
