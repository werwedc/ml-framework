namespace MLFramework.Communication.Backends;

using MLFramework.Communication;
using MLFramework.Communication.Async;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

/// <summary>
/// RCCL backend for AMD GPU communication
/// </summary>
public class RCCLBackend : IAsyncCommunicationBackend
{
    private readonly IntPtr _comm; // RCCL communicator handle
    private readonly int _rank;
    private readonly int _worldSize;
    private readonly CommunicationConfig _config;
    private readonly DeviceType _deviceType;
    private readonly IntPtr _stream; // ROCm stream
    private bool _disposed;

    public int Rank => _rank;
    public int WorldSize => _worldSize;
    public string BackendName => "RCCL";
    public DeviceType Device => _deviceType;

    /// <summary>
    /// Initialize RCCL backend
    /// </summary>
    /// <param name="rank">Rank of this process</param>
    /// <param name="worldSize">Total number of processes</param>
    /// <param name="config">Configuration</param>
    public RCCLBackend(int rank, int worldSize, CommunicationConfig config)
    {
        _rank = rank;
        _worldSize = worldSize;
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _deviceType = DeviceType.ROCm;

        // Create ROCm stream
        _stream = Native.RCCLNative.CreateHipStream();

        // Initialize RCCL communicator (via P/Invoke)
        _comm = InitializeRCCLComm(rank, worldSize);
    }

    /// <summary>
    /// Initialize RCCL communicator (P/Invoke wrapper)
    /// </summary>
    private IntPtr InitializeRCCLComm(int rank, int worldSize)
    {
        IntPtr comm;
        int result = Native.RCCLNative.rcclCommInitRank(
            out comm,
            worldSize,
            IntPtr.Zero, // commId - will use RCCL's internal ID generation
            rank);

        if (result != Native.RCCLNative.RCCL_SUCCESS)
        {
            throw new CommunicationException($"Failed to initialize RCCL communicator: {result}", _rank, BackendName);
        }

        return comm;
    }

    public void Broadcast(Tensor tensor, int rootRank)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for RCCL operations");

        // Call RCCL broadcast via P/Invoke
        RCCLBroadcast(tensor, rootRank);
    }

    public Tensor Reduce(Tensor tensor, ReduceOp operation, int rootRank)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for RCCL operations");

        return RCCLReduce(tensor, operation, rootRank);
    }

    public Tensor AllReduce(Tensor tensor, ReduceOp operation)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for RCCL operations");

        return RCCLAllReduce(tensor, operation);
    }

    public Tensor AllGather(Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for RCCL operations");

        return RCCLAllGather(tensor);
    }

    public Tensor ReduceScatter(Tensor tensor, ReduceOp operation)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for RCCL operations");

        return RCCLReduceScatter(tensor, operation);
    }

    public void Barrier()
    {
        RCCLBarrier();
    }

    // Async operations
    public ICommunicationHandle BroadcastAsync(Tensor tensor, int rootRank)
    {
        var task = Task.Run(() =>
        {
            Broadcast(tensor, rootRank);
            return tensor;
        });
        return new AsyncCommunicationHandle(task);
    }

    public ICommunicationHandle AllReduceAsync(Tensor tensor, ReduceOp operation)
    {
        var task = Task.Run(() => AllReduce(tensor, operation));
        return new AsyncCommunicationHandle(task);
    }

    public ICommunicationHandle BarrierAsync()
    {
        var task = Task.Run(() =>
        {
            Barrier();
            return Tensor.Zeros(new[] { 0 });
        });
        return new AsyncCommunicationHandle(task);
    }

    // RCCL P/Invoke methods
    private void RCCLBroadcast(Tensor tensor, int rootRank)
    {
        var dataPtr = GetGpuDataPointer(tensor);
        var count = tensor.Size;
        var datatype = Native.RCCLNative.GetRCCLDataType(tensor.Dtype);

        int result = Native.RCCLNative.rcclBroadcast(
            dataPtr,
            dataPtr,
            count,
            datatype,
            rootRank,
            _comm,
            _stream);

        if (result != Native.RCCLNative.RCCL_SUCCESS)
        {
            throw new CommunicationException($"RCCL broadcast failed: {Native.RCCLNative.GetErrorString(result)}", _rank, BackendName);
        }

        // Synchronize HIP stream
        Native.RCCLNative.HipStreamSynchronize(_stream);
    }

    private Tensor RCCLReduce(Tensor tensor, ReduceOp operation, int rootRank)
    {
        var result = Tensor.Zeros(tensor.Shape, tensor.Dtype);
        var sendPtr = GetGpuDataPointer(tensor);
        var recvPtr = GetGpuDataPointer(result);
        var count = tensor.Size;
        var datatype = Native.RCCLNative.GetRCCLDataType(tensor.Dtype);
        var rcclOp = MapReduceOp(operation);

        int rcclResult = Native.RCCLNative.rcclReduce(
            sendPtr,
            recvPtr,
            count,
            datatype,
            rcclOp,
            rootRank,
            _comm,
            _stream);

        if (rcclResult != Native.RCCLNative.RCCL_SUCCESS)
        {
            throw new CommunicationException($"RCCL reduce failed: {Native.RCCLNative.GetErrorString(rcclResult)}", _rank, BackendName);
        }

        // Synchronize HIP stream
        Native.RCCLNative.HipStreamSynchronize(_stream);

        return result;
    }

    private Tensor RCCLAllReduce(Tensor tensor, ReduceOp operation)
    {
        var result = Tensor.Zeros(tensor.Shape, tensor.Dtype);
        var sendPtr = GetGpuDataPointer(tensor);
        var recvPtr = GetGpuDataPointer(result);
        var count = tensor.Size;
        var datatype = Native.RCCLNative.GetRCCLDataType(tensor.Dtype);
        var rcclOp = MapReduceOp(operation);

        int rcclResult = Native.RCCLNative.rcclAllReduce(
            sendPtr,
            recvPtr,
            count,
            datatype,
            rcclOp,
            _comm,
            _stream);

        if (rcclResult != Native.RCCLNative.RCCL_SUCCESS)
        {
            throw new CommunicationException($"RCCL all-reduce failed: {Native.RCCLNative.GetErrorString(rcclResult)}", _rank, BackendName);
        }

        // Synchronize HIP stream
        Native.RCCLNative.HipStreamSynchronize(_stream);

        return result;
    }

    private Tensor RCCLAllGather(Tensor tensor)
    {
        int totalSize = tensor.Size * _worldSize;
        var result = Tensor.Zeros(new[] { totalSize }, tensor.Dtype);
        var sendPtr = GetGpuDataPointer(tensor);
        var recvPtr = GetGpuDataPointer(result);
        var count = tensor.Size;
        var datatype = Native.RCCLNative.GetRCCLDataType(tensor.Dtype);

        int rcclResult = Native.RCCLNative.rcclAllGather(
            sendPtr,
            recvPtr,
            count,
            datatype,
            _comm,
            _stream);

        if (rcclResult != Native.RCCLNative.RCCL_SUCCESS)
        {
            throw new CommunicationException($"RCCL all-gather failed: {Native.RCCLNative.GetErrorString(rcclResult)}", _rank, BackendName);
        }

        // Synchronize HIP stream
        Native.RCCLNative.HipStreamSynchronize(_stream);

        return result;
    }

    private Tensor RCCLReduceScatter(Tensor tensor, ReduceOp operation)
    {
        int chunkSize = tensor.Size / _worldSize;
        var result = Tensor.Zeros(new[] { chunkSize }, tensor.Dtype);
        var sendPtr = GetGpuDataPointer(tensor);
        var recvPtr = GetGpuDataPointer(result);
        var count = chunkSize;
        var datatype = Native.RCCLNative.GetRCCLDataType(tensor.Dtype);
        var rcclOp = MapReduceOp(operation);

        int rcclResult = Native.RCCLNative.rcclReduceScatter(
            sendPtr,
            recvPtr,
            count,
            datatype,
            rcclOp,
            _comm,
            _stream);

        if (rcclResult != Native.RCCLNative.RCCL_SUCCESS)
        {
            throw new CommunicationException($"RCCL reduce-scatter failed: {Native.RCCLNative.GetErrorString(rcclResult)}", _rank, BackendName);
        }

        // Synchronize HIP stream
        Native.RCCLNative.HipStreamSynchronize(_stream);

        return result;
    }

    private void RCCLBarrier()
    {
        int result = Native.RCCLNative.rcclBarrier(_comm, _stream);

        if (result != Native.RCCLNative.RCCL_SUCCESS)
        {
            throw new CommunicationException($"RCCL barrier failed: {Native.RCCLNative.GetErrorString(result)}", _rank, BackendName);
        }

        // Synchronize HIP stream
        Native.RCCLNative.HipStreamSynchronize(_stream);
    }

    private bool IsTensorOnGPU(Tensor tensor)
    {
        // RCCL backend assumes all tensors are on ROCm devices
        // In a real implementation, you would check the tensor's device property
        // For now, we assume tensors are on GPU since RCCL requires it
        return _deviceType == DeviceType.ROCm;
    }

    private int MapReduceOp(ReduceOp op)
    {
        return op switch
        {
            ReduceOp.Sum => Native.RCCLNative.RCCL_SUM,
            ReduceOp.Product => Native.RCCLNative.RCCL_PROD,
            ReduceOp.Max => Native.RCCLNative.RCCL_MAX,
            ReduceOp.Min => Native.RCCLNative.RCCL_MIN,
            ReduceOp.Avg => Native.RCCLNative.RCCL_AVG,
            _ => throw new ArgumentException($"Unsupported reduce operation: {op}")
        };
    }

    private IntPtr GetGpuDataPointer(Tensor tensor)
    {
        // Get HIP pointer to tensor data
        // In a real implementation, this would access the GPU memory pointer
        return IntPtr.Zero; // Placeholder
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_comm != IntPtr.Zero)
            {
                int result = Native.RCCLNative.rcclCommDestroy(_comm);
                if (result != Native.RCCLNative.RCCL_SUCCESS)
                {
                    // Log error but don't throw in Dispose
                    Console.WriteLine($"Warning: Failed to destroy RCCL communicator: {result}");
                }
            }

            if (_stream != IntPtr.Zero)
            {
                Native.RCCLNative.DestroyHipStream(_stream);
            }

            _disposed = true;
        }
    }
}

/// <summary>
/// RCCL reduce operations
/// </summary>
internal enum RCCLReduceOperation
{
    RCCL_SUM = 0,
    RCCL_PROD = 1,
    RCCL_MAX = 2,
    RCCL_MIN = 3,
    RCCL_AVG = 4
}
