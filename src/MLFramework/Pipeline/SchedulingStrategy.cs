namespace MLFramework.Pipeline
{
    /// <summary>
    /// Pipeline scheduling strategy
    /// </summary>
    public enum SchedulingStrategy
    {
        /// <summary>
        /// Classic GPipe: fill -> steady state -> drain
        /// </summary>
        GPipe,

        /// <summary>
        /// Interleaved 1F1B (PipeDream-Flush) - to be implemented later
        /// </summary>
        Interleaved1F1B
    }
}
