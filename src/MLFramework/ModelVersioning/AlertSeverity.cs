namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Severity level for version alerts.
    /// </summary>
    public enum AlertSeverity
    {
        /// <summary>
        /// Informational alert, no immediate action required.
        /// </summary>
        Info,

        /// <summary>
        /// Warning alert, monitor closely.
        /// </summary>
        Warning,

        /// <summary>
        /// Critical alert, immediate attention required.
        /// </summary>
        Critical
    }
}
