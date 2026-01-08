using System;

namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Statistics for communication profiles
    /// </summary>
    public class CommunicationProfileStatistics
    {
        public int TotalOperations { get; set; }
        public long TotalDataTransferred { get; set; }
        public double TotalTime { get; set; }
        public double AverageBandwidth { get; set; }
        public double MinBandwidth { get; set; }
        public double MaxBandwidth { get; set; }

        public override string ToString()
        {
            return $"Total Operations: {TotalOperations}, " +
                   $"Total Data: {TotalDataTransferred / 1024.0 / 1024.0 / 1024.0:F2} GB, " +
                   $"Total Time: {TotalTime / 1000.0:F2} s, " +
                   $"Avg Bandwidth: {AverageBandwidth:F2} MB/s";
        }
    }
}
