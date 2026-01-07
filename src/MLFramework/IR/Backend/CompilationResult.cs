namespace MLFramework.IR.Backend
{
    /// <summary>
    /// Result of compilation
    /// </summary>
    public class CompilationResult
    {
        /// <summary>
        /// Whether compilation was successful
        /// </summary>
        public bool Success { get; set; } = true;

        /// <summary>
        /// Generated source code
        /// </summary>
        public string Code { get; set; }

        /// <summary>
        /// Generated binary
        /// </summary>
        public byte[] Binary { get; set; }

        /// <summary>
        /// Error message if compilation failed
        /// </summary>
        public string ErrorMessage { get; set; }

        /// <summary>
        /// Get a string representation of the result
        /// </summary>
        public override string ToString()
        {
            if (Success)
            {
                return $"Compilation succeeded. Code length: {Code?.Length ?? 0} bytes";
            }
            else
            {
                return $"Compilation failed: {ErrorMessage}";
            }
        }
    }
}
