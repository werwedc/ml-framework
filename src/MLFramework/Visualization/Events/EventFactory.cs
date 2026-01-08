using System.Diagnostics;

namespace MachineLearning.Visualization.Events;

/// <summary>
/// Factory for creating common event types with auto-generated timestamps and step numbers
/// </summary>
public static class EventFactory
{
    private static long _globalStep = 0;
    private static readonly object _stepLock = new object();

    /// <summary>
    /// Gets or sets global step counter
    /// </summary>
    public static long GlobalStep
    {
        get
        {
            lock (_stepLock)
            {
                return _globalStep;
            }
        }
        set
        {
            lock (_stepLock)
            {
                _globalStep = value;
            }
        }
    }

    /// <summary>
    /// Increments and returns next global step number
    /// </summary>
    public static long NextStep()
    {
        lock (_stepLock)
        {
            return ++_globalStep;
        }
    }

    /// <summary>
    /// Creates a scalar metric event with auto-generated step number if not provided
    /// </summary>
    /// <param name="name">Name of the metric</param>
    /// <param name="value">Value of the metric</param>
    /// <param name="step">Training step (uses global counter if -1)</param>
    /// <param name="metadata">Optional metadata</param>
    public static ScalarMetricEvent CreateScalar(string name, float value, long step = -1, Dictionary<string, string>? metadata = null)
    {
        var actualStep = step == -1 ? NextStep() : step;
        return new ScalarMetricEvent(name, value, actualStep, metadata);
    }

    /// <summary>
    /// Creates a histogram event with auto-generated step number if not provided
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of histogram values</param>
    /// <param name="step">Training step (uses global counter if -1)</param>
    /// <param name="binCount">Number of bins</param>
    /// <param name="useLogScale">Whether to use log scale</param>
    /// <param name="min">Minimum value for binning</param>
    /// <param name="max">Maximum value for binning</param>
    /// <param name="metadata">Optional metadata</param>
    public static HistogramEvent CreateHistogram(
        string name,
        float[] values,
        long step = -1,
        int binCount = 30,
        bool useLogScale = false,
        float min = float.MinValue,
        float max = float.MaxValue,
        Dictionary<string, string>? metadata = null)
    {
        var actualStep = step == -1 ? NextStep() : step;
        return new HistogramEvent(name, values, actualStep, binCount, useLogScale, min, max, metadata);
    }

    /// <summary>
    /// Creates a profiling start event with auto-generated step number if not provided
    /// </summary>
    /// <param name="name">Name of the operation</param>
    /// <param name="step">Training step (uses global counter if -1)</param>
    /// <param name="metadata">Optional metadata</param>
    public static ProfilingStartEvent CreateProfilingStart(string name, long step = -1, Dictionary<string, string>? metadata = null)
    {
        var actualStep = step == -1 ? NextStep() : step;
        return new ProfilingStartEvent(name, actualStep, metadata);
    }

    /// <summary>
    /// Creates a profiling end event with auto-generated step number if not provided
    /// </summary>
    /// <param name="name">Name of the operation</param>
    /// <param name="step">Training step (uses global counter if -1)</param>
    /// <param name="durationNanoseconds">Duration of the operation</param>
    /// <param name="metadata">Optional metadata</param>
    public static ProfilingEndEvent CreateProfilingEnd(string name, long step = -1, long durationNanoseconds = 0, Dictionary<string, string>? metadata = null)
    {
        var actualStep = step == -1 ? NextStep() : step;
        return new ProfilingEndEvent(name, actualStep, durationNanoseconds, metadata);
    }

    /// <summary>
    /// Creates profiling start and end events with measured duration
    /// </summary>
    /// <param name="name">Name of the operation</param>
    /// <param name="step">Training step (uses global counter if -1)</param>
    /// <param name="action">Action to profile</param>
    /// <param name="metadata">Optional metadata</param>
    /// <returns>Result of action</returns>
    public static (ProfilingStartEvent startEvent, ProfilingEndEvent endEvent) CreateProfilingPair(
        string name,
        Action action,
        long step = -1,
        Dictionary<string, string>? metadata = null)
    {
        if (action == null)
            throw new ArgumentNullException(nameof(action));

        var actualStep = step == -1 ? NextStep() : step;
        var startEvent = new ProfilingStartEvent(name, actualStep, metadata);

        var stopwatch = Stopwatch.StartNew();
        var endEvent = new ProfilingEndEvent(name, actualStep, 0, metadata);
        try
        {
            action();
        }
        finally
        {
            stopwatch.Stop();
            var durationNanoseconds = stopwatch.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);
            endEvent = new ProfilingEndEvent(name, actualStep, durationNanoseconds, metadata);
        }
        return (startEvent, endEvent);
    }

    /// <summary>
    /// Creates profiling start and end events with measured duration for async actions
    /// </summary>
    /// <param name="name">Name of the operation</param>
    /// <param name="func">Async function to profile</param>
    /// <param name="step">Training step (uses global counter if -1)</param>
    /// <param name="metadata">Optional metadata</param>
    /// <returns>Result of action and profiling events</returns>
    public static async Task<(ProfilingStartEvent startEvent, ProfilingEndEvent endEvent)> CreateProfilingPairAsync<T>(
        string name,
        Func<Task<T>> func,
        long step = -1,
        Dictionary<string, string>? metadata = null)
    {
        if (func == null)
            throw new ArgumentNullException(nameof(func));

        var actualStep = step == -1 ? NextStep() : step;
        var startEvent = new ProfilingStartEvent(name, actualStep, metadata);

        var stopwatch = Stopwatch.StartNew();
        var endEvent = new ProfilingEndEvent(name, actualStep, 0, metadata);
        try
        {
            await func();
        }
        finally
        {
            stopwatch.Stop();
            var durationNanoseconds = stopwatch.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);
            endEvent = new ProfilingEndEvent(name, actualStep, durationNanoseconds, metadata);
        }
        return (startEvent, endEvent);
    }

    /// <summary>
    /// Creates a memory allocation event with auto-generated step number if not provided
    /// </summary>
    /// <param name="name">Name of the allocation</param>
    /// <param name="sizeBytes">Size in bytes</param>
    /// <param name="step">Training step (uses global counter if -1)</param>
    /// <param name="location">Memory location</param>
    /// <param name="metadata">Optional metadata</param>
    public static MemoryAllocationEvent CreateMemoryAllocation(string name, long sizeBytes, long step = -1, string location = "CPU", Dictionary<string, string>? metadata = null)
    {
        var actualStep = step == -1 ? NextStep() : step;
        return new MemoryAllocationEvent(name, sizeBytes, actualStep, location, metadata);
    }

    /// <summary>
    /// Creates a tensor operation event with auto-generated step number if not provided
    /// </summary>
    /// <param name="operationName">Name of the operation</param>
    /// <param name="inputShapes">Shapes of input tensors</param>
    /// <param name="outputShape">Shape of output tensor</param>
    /// <param name="step">Training step (uses global counter if -1)</param>
    /// <param name="durationNanoseconds">Duration of the operation</param>
    /// <param name="metadata">Optional metadata</param>
    public static TensorOperationEvent CreateTensorOperation(
        string operationName,
        int[][] inputShapes,
        int[] outputShape,
        long step = -1,
        long durationNanoseconds = 0,
        Dictionary<string, string>? metadata = null)
    {
        var actualStep = step == -1 ? NextStep() : step;
        return new TensorOperationEvent(operationName, inputShapes, outputShape, actualStep, durationNanoseconds, metadata);
    }
}
