using System;
using MLFramework.Communication;
using MLFramework.Distributed.Communication;

namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Selects optimal communication algorithm
    /// </summary>
    public class AlgorithmSelector
    {
        private readonly int _worldSize;
        private readonly CommunicationConfig _config;

        public AlgorithmSelector(int worldSize, CommunicationConfig config)
        {
            _worldSize = worldSize;
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Select optimal algorithm for all-reduce
        /// </summary>
        public CommunicationAlgorithm SelectAllReduceAlgorithm(long dataSizeBytes)
        {
            // Automatic selection based on data size and world size

            // For small messages, use recursive doubling
            if (dataSizeBytes < 4096 && _worldSize <= 8)
            {
                return CommunicationAlgorithm.RecursiveDoubling;
            }

            // For medium messages, use Rabenseifner
            if (dataSizeBytes < 1024 * 1024 && _worldSize <= 16)
            {
                return CommunicationAlgorithm.Rabenseifner;
            }

            // For large messages, use ring
            if (dataSizeBytes < 16 * 1024 * 1024)
            {
                return CommunicationAlgorithm.Ring;
            }

            // For very large messages, use tree
            return CommunicationAlgorithm.Tree;
        }

        /// <summary>
        /// Select optimal algorithm for all-gather
        /// </summary>
        public CommunicationAlgorithm SelectAllGatherAlgorithm(long dataSizeBytes)
        {
            // Ring is generally optimal for all-gather
            return CommunicationAlgorithm.Ring;
        }

        /// <summary>
        /// Select optimal algorithm for reduce-scatter
        /// </summary>
        public CommunicationAlgorithm SelectReduceScatterAlgorithm(long dataSizeBytes)
        {
            // Rabenseifner is optimal for reduce-scatter
            return CommunicationAlgorithm.Rabenseifner;
        }

        /// <summary>
        /// Get algorithm name
        /// </summary>
        public static string GetAlgorithmName(CommunicationAlgorithm algorithm)
        {
            return algorithm switch
            {
                CommunicationAlgorithm.Ring => "Ring",
                CommunicationAlgorithm.Tree => "Tree",
                CommunicationAlgorithm.RecursiveDoubling => "RecursiveDoubling",
                CommunicationAlgorithm.Rabenseifner => "Rabenseifner",
                CommunicationAlgorithm.Automatic => "Automatic",
                _ => "Unknown"
            };
        }
    }
}
