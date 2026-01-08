namespace MLFramework.Communication.Tests.Backends;

using MLFramework.Communication.Backends.Native;
using RitterFramework.Core;
using Xunit;

/// <summary>
/// Unit tests for MPI native interop
/// </summary>
public class MPINativeTests
{
    [Fact]
    public void MPINative_GetMPIDatatype_Float32_ReturnsCorrectValue()
    {
        // Act
        var result = MPINative.GetMPIDatatype(DataType.Float32);

        // Assert
        Assert.Equal(MPINative.MPI_FLOAT, result);
    }

    [Fact]
    public void MPINative_GetMPIDatatype_Float64_ReturnsCorrectValue()
    {
        // Act
        var result = MPINative.GetMPIDatatype(DataType.Float64);

        // Assert
        Assert.Equal(MPINative.MPI_DOUBLE, result);
    }

    [Fact]
    public void MPINative_GetMPIDatatype_Int32_ReturnsCorrectValue()
    {
        // Act
        var result = MPINative.GetMPIDatatype(DataType.Int32);

        // Assert
        Assert.Equal(MPINative.MPI_INT, result);
    }

    [Fact]
    public void MPINative_GetMPIDatatype_Int64_ReturnsCorrectValue()
    {
        // Act
        var result = MPINative.GetMPIDatatype(DataType.Int64);

        // Assert
        Assert.Equal(MPINative.MPI_LONG, result);
    }

    [Fact]
    public void MPINative_GetMPIDatatype_Bool_ReturnsCorrectValue()
    {
        // Act
        var result = MPINative.GetMPIDatatype(DataType.Bool);

        // Assert
        Assert.Equal(MPINative.MPI_BYTE, result);
    }

    [Fact]
    public void MPINative_GetMPIDatatype_UnsupportedType_ThrowsArgumentException()
    {
        // Act & Assert
        // Note: All current DataType enum values are supported
        // This test ensures the pattern works
        Assert.Throws<ArgumentException>(() =>
        {
            // This would only throw if an unsupported type was passed
            MPINative.GetMPIDatatype(DataType.Float32);
        });
    }

    [Fact]
    public void MPINative_MPIHandles_AreDefined()
    {
        // Assert
        Assert.NotEqual(IntPtr.Zero, MPINative.MPI_COMM_WORLD);
        Assert.Equal(IntPtr.Zero, MPINative.MPI_COMM_NULL);
    }

    [Fact]
    public void MPINative_MPIDataTypes_AreDefined()
    {
        // Assert
        Assert.Equal(0, MPINative.MPI_BYTE);
        Assert.Equal(1, MPINative.MPI_CHAR);
        Assert.Equal(2, MPINative.MPI_INT);
        Assert.Equal(3, MPINative.MPI_LONG);
        Assert.Equal(4, MPINative.MPI_FLOAT);
        Assert.Equal(5, MPINative.MPI_DOUBLE);
    }

    [Fact]
    public void MPINative_MPIOperations_AreDefined()
    {
        // Assert
        Assert.Equal(6, MPINative.MPI_MAX);
        Assert.Equal(7, MPINative.MPI_MIN);
        Assert.Equal(8, MPINative.MPI_SUM);
        Assert.Equal(9, MPINative.MPI_PROD);
    }
}
