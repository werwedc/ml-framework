using System;

namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Profile data for a communication operation
    /// </summary>
    public class CommunicationProfile
    {
        public string Operation { get; set; } = "";
        public long DataSizeBytes { get; set; }
        public TimeSpan Duration { get; set; }
        public double BandwidthMBps { get; set; }
        public int NumRanks { get; set; }
        public string Algorithm { get; set; } = "";
        public DateTime Timestamp { get; set; }

        public override string ToString()
        {
            return $"{Operation} ({Algorithm}): {DataSizeBytes / 1024.0 / 1024.0:F2} MB, " +
                   $"{Duration.TotalMilliseconds:F2} ms, " +
                   $"{BandwidthMBps:F2} MB/s, " +
                   $"{NumRanks} ranks";
        }
    }
}
