namespace MLFramework.ModelZoo.Plugins
{
    /// <summary>
    /// Attribute to mark classes as registry plugins for auto-discovery.
    /// </summary>
    [System.AttributeUsage(System.AttributeTargets.Class)]
    public class RegistryPluginAttribute : System.Attribute
    {
        /// <summary>
        /// Gets the registry name.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the plugin priority. Higher values are checked first.
        /// </summary>
        public int Priority { get; }

        /// <summary>
        /// Gets the optional assembly name to load the plugin from.
        /// If null, the current assembly is used.
        /// </summary>
        public string AssemblyName { get; set; }

        /// <summary>
        /// Initializes a new instance of the RegistryPluginAttribute.
        /// </summary>
        /// <param name="name">The registry name.</param>
        /// <param name="priority">The plugin priority (default: 0).</param>
        public RegistryPluginAttribute(string name, int priority = 0)
        {
            Name = name ?? throw new System.ArgumentNullException(nameof(name));
            Priority = priority;
        }
    }

    /// <summary>
    /// Attribute to mark configuration classes for plugins.
    /// </summary>
    [System.AttributeUsage(System.AttributeTargets.Class)]
    public class RegistryPluginConfigAttribute : System.Attribute
    {
        /// <summary>
        /// Gets the registry name this configuration is for.
        /// </summary>
        public string RegistryName { get; }

        /// <summary>
        /// Initializes a new instance of the RegistryPluginConfigAttribute.
        /// </summary>
        /// <param name="registryName">The registry name.</param>
        public RegistryPluginConfigAttribute(string registryName)
        {
            RegistryName = registryName ?? throw new System.ArgumentNullException(nameof(registryName));
        }
    }
}
