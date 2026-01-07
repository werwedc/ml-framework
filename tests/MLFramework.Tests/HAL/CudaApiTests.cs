using Xunit;
using MLFramework.HAL.CUDA;

namespace MLFramework.HAL.Tests;

public class CudaApiTests
{
    [Fact]
    public void CudaGetDeviceCount_ReturnsNonNegative()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var result = CudaApi.CudaGetDeviceCount(out int count);
        CudaException.CheckError(result);

        Assert.True(count >= 0);
    }

    [Fact]
    public void CudaMalloc_Free_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var allocResult = CudaApi.CudaMalloc(out IntPtr ptr, 1024);
        CudaException.CheckError(allocResult);

        Assert.NotEqual(IntPtr.Zero, ptr);

        var freeResult = CudaApi.CudaFree(ptr);
        CudaException.CheckError(freeResult);
    }

    [Fact]
    public void CudaGetDeviceProperties_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var countResult = CudaApi.CudaGetDeviceCount(out int count);
        CudaException.CheckError(countResult);

        Assert.True(count > 0, "Expected at least one CUDA device");

        var props = new CudaDeviceProperties();
        var propsResult = CudaApi.CudaGetDeviceProperties(ref props, 0);
        CudaException.CheckError(propsResult);

        Assert.NotNull(props.Name);
        Assert.True(props.TotalGlobalMem > 0);
    }

    [Fact]
    public void CudaSetDevice_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var result = CudaApi.CudaGetDeviceCount(out int count);
        CudaException.CheckError(result);

        if (count > 0)
        {
            var setResult = CudaApi.CudaSetDevice(0);
            CudaException.CheckError(setResult);
        }
    }

    [Fact]
    public void CudaMemset_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var allocResult = CudaApi.CudaMalloc(out IntPtr ptr, 1024);
        CudaException.CheckError(allocResult);

        var memsetResult = CudaApi.CudaMemset(ptr, 0, 1024);
        CudaException.CheckError(memsetResult);

        var freeResult = CudaApi.CudaFree(ptr);
        CudaException.CheckError(freeResult);
    }

    [Fact]
    public void CudaStreamCreate_Destroy_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var result = CudaApi.CudaStreamCreate(out IntPtr stream);
        CudaException.CheckError(result);

        Assert.NotEqual(IntPtr.Zero, stream);

        var destroyResult = CudaApi.CudaStreamDestroy(stream);
        CudaException.CheckError(destroyResult);
    }

    [Fact]
    public void CudaEventCreate_Destroy_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var result = CudaApi.CudaEventCreate(out IntPtr evt);
        CudaException.CheckError(result);

        Assert.NotEqual(IntPtr.Zero, evt);

        var destroyResult = CudaApi.CudaEventDestroy(evt);
        CudaException.CheckError(destroyResult);
    }

    [Fact]
    public void CudaEventQuery_NotReady_BeforeRecord()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var result = CudaApi.CudaEventCreate(out IntPtr evt);
        CudaException.CheckError(result);

        try
        {
            var queryResult = CudaApi.CudaEventQuery(evt);

            // Event should not be ready until recorded
            Assert.Equal(CudaError.NotReady, queryResult);
        }
        finally
        {
            CudaApi.CudaEventDestroy(evt);
        }
    }

    [Fact]
    public void CudaEventRecord_Query_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var createResult = CudaApi.CudaStreamCreate(out IntPtr stream);
        CudaException.CheckError(createResult);

        var evtResult = CudaApi.CudaEventCreate(out IntPtr evt);
        CudaException.CheckError(evtResult);

        try
        {
            // Record event in stream
            var recordResult = CudaApi.CudaEventRecord(evt, stream);
            CudaException.CheckError(recordResult);

            // Sync stream to ensure event is complete
            var syncResult = CudaApi.CudaStreamSynchronize(stream);
            CudaException.CheckError(syncResult);

            // Query event should now return success
            var queryResult = CudaApi.CudaEventQuery(evt);
            Assert.Equal(CudaError.Success, queryResult);
        }
        finally
        {
            CudaApi.CudaEventDestroy(evt);
            CudaApi.CudaStreamDestroy(stream);
        }
    }

    private bool CudaAvailable()
    {
        try
        {
            var result = CudaApi.CudaGetDeviceCount(out int count);
            return result == CudaError.Success && count > 0;
        }
        catch (DllNotFoundException)
        {
            // CUDA DLL not available
            return false;
        }
    }
}
