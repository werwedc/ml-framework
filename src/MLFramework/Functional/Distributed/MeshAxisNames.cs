namespace MLFramework.Functional.Distributed
{
    public static class MeshAxisNames
    {
        /// <summary>
        /// Data/batch parallelism axis.
        /// </summary>
        public const string Data = "data";

        /// <summary>
        /// Model parallelism axis (e.g., for tensor parallelism).
        /// </summary>
        public const string Model = "model";

        /// <summary>
        /// Pipeline parallelism axis.
        /// </summary>
        public const string Pipeline = "pipeline";
    }
}
