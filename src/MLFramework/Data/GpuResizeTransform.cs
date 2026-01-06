namespace MLFramework.Data;

/// <summary>
/// GPU-accelerated image resize transform.
/// </summary>
public class GpuResizeTransform : IGpuTransform
{
    private readonly int _width;
    private readonly int _height;
    private readonly InterpolationMode _mode;
    private int _gpuDevice;

    /// <summary>
    /// Initializes a new instance of the GpuResizeTransform class.
    /// </summary>
    /// <param name="width">Target width.</param>
    /// <param name="height">Target height.</param>
    /// <param name="mode">Interpolation mode.</param>
    public GpuResizeTransform(int width, int height, InterpolationMode mode = InterpolationMode.Bilinear)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Width and height must be positive.");

        _width = width;
        _height = height;
        _mode = mode;
        _gpuDevice = 0;
    }

    /// <inheritdoc/>
    public bool GpuAvailable => DetectGpuSupport();

    /// <inheritdoc/>
    public int GpuDevice => _gpuDevice;

    /// <inheritdoc/>
    public void SetGpuDevice(int deviceId)
    {
        _gpuDevice = deviceId;
    }

    /// <inheritdoc/>
    public object Apply(object input)
    {
        if (input is float[,] cpuImage)
        {
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
