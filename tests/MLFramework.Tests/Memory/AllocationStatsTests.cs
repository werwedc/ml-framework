using Xunit;

namespace MLFramework.Tests.Memory;

public class AllocationStatsTests
{
    [Fact]
    public void Constructor_InitializesWithDefaultValues()
    {
        var stats = new AllocationStats();

        Assert.Equal(0, stats.TotalAllocations);
        Assert.Equal(0, stats.TotalResizes);
        Assert.Equal(0, stats.TotalBytesAllocated);
        Assert.Equal(0, stats.TotalBytesWasted);
        Assert.Equal(0.0, stats.AverageUtilization);
    }

    [Fact]
    public void ToReport_GeneratesReadableReport()
    {
        var stats = new AllocationStats
        {
            TotalAllocations = 100,
            TotalResizes = 25,
            TotalBytesAllocated = 1000000,
            TotalBytesWasted = 150000,
            AverageUtilization = 0.85
        };

        var report = stats.ToReport();

        Assert.Contains("Total Allocations:     100", report);
        Assert.Contains("Total Resizes:         25", report);
        Assert.Contains("Total Bytes Allocated: 1,000,000", report);
        Assert.Contains("Total Bytes Wasted:    150,000", report);
        Assert.Contains("85.00%", report);
    }

    [Fact]
    public void ToReport_HandlesZeroValues()
    {
        var stats = new AllocationStats();

        var report = stats.ToReport();

        Assert.Contains("0.00%", report);
    }
}
