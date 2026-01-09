namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Deployment environment types for models.
    /// </summary>
    public enum DeploymentEnv
    {
        /// <summary>
        /// Cloud deployment with abundant resources.
        /// </summary>
        Cloud,

        /// <summary>
        /// On-premises deployment.
        /// </summary>
        OnPremises,

        /// <summary>
        /// Edge deployment (IoT devices, gateways).
        /// </summary>
        Edge,

        /// <summary>
        /// Mobile device deployment (phones, tablets).
        /// </summary>
        Mobile,

        /// <summary>
        /// Browser-based deployment.
        /// </summary>
        Web
    }
}
