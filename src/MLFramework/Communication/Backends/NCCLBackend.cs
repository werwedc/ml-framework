namespace MLFramework.Communication.Backends;

using MLFramework.Communication;
using MLFramework.Communication.Async;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

/// <summary>
/// NCCL backend for NVIDIA GPU communication
/// </summary>
public class NCCLBackend : IAsyncCommunicationBackend
{
    private readonly IntPtr _comm; // NCCL communicator handle
    private readonly int _rank;
    private readonly int _worldSize;
    private readonly CommunicationConfig _config;
    private readonly DeviceType _deviceType;
    private readonly IntPtr _stream; // CUDA stream
    private bool _disposed;

    public int Rank => _rank;
    public int WorldSize => _worldSize;
    public string BackendName => "NCCL";
    public DeviceType Device => _deviceType;

    /// <summary>
    /// Initialize NCCL backend
    /// </summary>
    /// <param name="rank">Rank of this process</param>
    /// <param name="worldSize">Total number of processes</param>
    /// <param name="config">Configuration</param>
    public NCCLBackend(int rank, int worldSize, CommunicationConfig config)
    {
        _rank = rank;
        _worldSize = worldSize;
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _deviceType = DeviceType.CUDA;

        // Create CUDA stream
        _stream = Native.NCCLNative.CreateCudaStream();

        // Initialize NCCL communicator (via P/Invoke)
        _comm = InitializeNCCLComm(rank, worldSize);
    }

    /// <summary>
    /// Initialize NCCL communicator (P/Invoke wrapper)
    /// </summary>
    private IntPtr InitializeNCCLComm(int rank, int worldSize)
    {
        IntPtr comm;
        int result = Native.NCCLNative.ncclCommInitRank(
            out comm,
            worldSize,
            IntPtr.Zero, // commId - will use NCCL's internal ID generation
            rank);

        if (result != Native.NCCLNative.NCCL_SUCCESS)
        {
            throw new CommunicationException($"Failed to initialize NCCL communicator: {result}", _rank, BackendName);
        }

        return comm;
    }

    public void Broadcast(Tensor tensor, int rootRank)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for NCCL operations");

        // Call NCCL broadcast via P/Invoke
        NCCLBroadcast(tensor, rootRank);
    }

    public Tensor Reduce(Tensor tensor, ReduceOp operation, int rootRank)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for NCCL operations");

        return NCCLReduce(tensor, operation, rootRank);
    }

    public Tensor AllReduce(Tensor tensor, ReduceOp operation)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for NCCL operations");

        return NCCLAllReduce(tensor, operation);
    }

    public Tensor AllGather(Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for NCCL operations");

        return NCCLAllGather(tensor);
    }

    public Tensor ReduceScatter(Tensor tensor, ReduceOp operation)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!IsTensorOnGPU(tensor))
            throw new ArgumentException("Tensor must be on GPU for NCCL operations");

        return NCCLReduceScatter(tensor, operation);
    }

    public void Barrier()
    {
        NCCLBarrier();
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

    // NCCL P/Invoke methods
    private void NCCLBroadcast(Tensor tensor, int rootRank)
    {
        var dataPtr = GetGpuDataPointer(tensor);
        var count = tensor.Size;
        var datatype = Native.NCCLNative.GetNCCLDataType(tensor.Dtype);

        int result = Native.NCCLNative.ncclBroadcast(
            dataPtr,
            dataPtr,
            count,
            datatype,
            rootRank,
            _comm,
            _stream);

        if (result != Native.NCCLNative.NCCL_SUCCESS)
        {
            throw new CommunicationException($"NCCL broadcast failed: {result}", _rank, BackendName);
        }

        // Synchronize CUDA stream
        Native.NCCLNative.CudaStreamSynchronize(_stream);
    }

    private Tensor NCCLReduce(Tensor tensor, ReduceOp operation, int rootRank)
    {
        var result = Tensor.Zeros(tensor.Shape, tensor.Dtype);
        var sendPtr = GetGpuDataPointer(tensor);
        var recvPtr = GetGpuDataPointer(result);
        var count = tensor.Size;
        var datatype = Native.NCCLNative.GetNCCLDataType(tensor.Dtype);
        var ncclOp = MapReduceOp(operation);

        int ncclResult = Native.NCCLNative.ncclReduce(
            sendPtr,
            recvPtr,
            count,
            datatype,
            ncclOp,
            rootRank,
            _comm,
            _stream);

        if (ncclResult != Native.NCCLNative.NCCL_SUCCESS)
        {
            throw new CommunicationException($"NCCL reduce failed: {ncclResult}", _rank, BackendName);
        }

        // Synchronize CUDA stream
        Native.NCCLNative.CudaStreamSynchronize(_stream);

        return result;
    }

    private Tensor NCCLAllReduce(Tensor tensor, ReduceOp operation)
    {
        var result = Tensor.Zeros(tensor.Shape, tensor.Dtype);
        var sendPtr = GetGpuDataPointer(tensor);
        var recvPtr = GetGpuDataPointer(result);
        var count = tensor.Size;
        var datatype = Native.NCCLNative.GetNCCLDataType(tensor.Dtype);
        var ncclOp = MapReduceOp(operation);

        int ncclResult = Native.NCCLNative.ncclAllReduce(
            sendPtr,
            recvPtr,
            count,
            datatype,
            ncclOp,
            _comm,
            _stream);

        if (ncclResult != Native.NCCLNative.NCCL_SUCCESS)
        {
            throw new CommunicationException($"NCCL all-reduce failed: {ncclResult}", _rank, BackendName);
        }

        // Synchronize CUDA stream
        Native.NCCLNative.CudaStreamSynchronize(_stream);

        return result;
    }

    private Tensor NCCLAllGather(Tensor tensor)
    {
        int totalSize = tensor.Size * _worldSize;
        var result = Tensor.Zeros(new[] { totalSize }, tensor.Dtype);
        var sendPtr = GetGpuDataPointer(tensor);
        var recvPtr = GetGpuDataPointer(result);
        var count = tensor.Size;
        var datatype = Native.NCCLNative.GetNCCLDataType(tensor.Dtype);

        int ncclResult = Native.NCCLNative.ncclAllGather(
            sendPtr,
            recvPtr,
            count,
            datatype,
            _comm,
            _stream);

        if (ncclResult != Native.NCCLNative.NCCL_SUCCESS)
        {
            throw new CommunicationException($"NCCL all-gather failed: {ncclResult}", _rank, BackendName);
        }

        // Synchronize CUDA stream
        Native.NCCLNative.CudaStreamSynchronize(_stream);

        return result;
    }

    private Tensor NCCLReduceScatter(Tensor tensor, ReduceOp operation)
    {
        int chunkSize = tensor.Size / _worldSize;
        var result = Tensor.Zeros(new[] { chunkSize }, tensor.Dtype);
        var sendPtr = GetGpuDataPointer(tensor);
        var recvPtr = GetGpuDataPointer(result);
        var count = chunkSize;
        var datatype = Native.NCCLNative.GetNCCLDataType(tensor.Dtype);
        var ncclOp = MapReduceOp(operation);

        int ncclResult = Native.NCCLNative.ncclReduceScatter(
            sendPtr,
            recvPtr,
            count,
            datatype,
            ncclOp,
            _comm,
            _stream);

        if (ncclResult != Native.NCCLNative.NCCL_SUCCESS)
        {
            throw new CommunicationException($"NCCL reduce-scatter failed: {ncclResult}", _rank, BackendName);
        }

        // Synchronize CUDA stream
        Native.NCCLNative.CudaStreamSynchronize(_stream);

        return result;
    }

    private void NCCLBarrier()
    {
        int result = Native.NCCLNative.ncclBarrier(_comm, _stream);

        if (result != Native.NCCLNative.NCCL_SUCCESS)
        {
            throw new CommunicationException($"NCCL barrier failed: {result}", _rank, BackendName);
        }

        // Synchronize CUDA stream
        Native.NCCLNative.CudaStreamSynchronize(_stream);
    }

    private bool IsTensorOnGPU(Tensor tensor)
    {
        // NCCL backend assumes all tensors are on CUDA devices
        // In a real implementation, you would check the tensor's device property
        // For now, we assume tensors are on GPU since NCCL requires it
        return _deviceType == DeviceType.CUDA;
    }

    private int MapReduceOp(ReduceOp op)
    {
        return op switch
        {
            ReduceOp.Sum => Native.NCCLNative.NCCL_SUM,
            ReduceOp.Product => Native.NCCLNative.NCCL_PROD,
            ReduceOp.Max => Native.NCCLNative.NCCL_MAX,
            ReduceOp.Min => Native.NCCLNative.NCCL_MIN,
            _ => throw new ArgumentException($"Unsupported reduce operation: {op}")
        };
    }

    private IntPtr GetGpuDataPointer(Tensor tensor)
    {
        // Get CUDA pointer to tensor data
        // In a real implementation, this would access the GPU memory pointer
        return IntPtr.Zero; // Placeholder
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_comm != IntPtr.Zero)
            {
                int result = Native.NCCLNative.ncclCommDestroy(_comm);
                if (result != Native.NCCLNative.NCCL_SUCCESS)
                {
                    // Log error but don't throw in Dispose
                    Console.WriteLine($"Warning: Failed to destroy NCCL communicator: {result}");
                }
            }

            if (_stream != IntPtr.Zero)
            {
                Native.NCCLNative.DestroyCudaStream(_stream);
            }

            _disposed = true;
        }
    }
}

/// <summary>
/// NCCL reduce operations
/// </summary>
internal enum NCCLReduceOperation
{
    NCCL_SUM = 0,
    NCCL_PROD = 1,
    NCCL_MAX = 2,
    NCCL_MIN = 3
}
