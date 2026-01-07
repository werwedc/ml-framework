namespace MLFramework.Pipeline
{
    /// <summary>
    /// Strategy for partitioning layers across pipeline stages
    /// </summary>
    public enum PartitionMode
    {
        /// <summary>
        /// Automatically partition based on memory and computation cost
        /// </summary>
        Automatic,

        /// <summary>
        /// User specifies which layers belong to each stage
        /// </summary>
        Manual,

        /// <summary>
        /// Evenly distribute layers across stages
        /// </summary>
        Uniform
    }
}
