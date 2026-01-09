using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Loader;

namespace MLFramework.ModelZoo.Plugins
{
    /// <summary>
    /// Discovers and loads plugins automatically.
    /// Supports assembly scanning, plugin directory loading, and hot-reloading.
    /// </summary>
    public class PluginDiscovery
    {
        private readonly PluginManager _pluginManager;
        private readonly List<Assembly> _loadedAssemblies;
        private readonly Dictionary<string, FileSystemWatcher> _watchers;
        private readonly object _lock = new object();

        /// <summary>
        /// Initializes a new instance of the PluginDiscovery.
        /// </summary>
        /// <param name="pluginManager">The plugin manager to register discovered plugins with.</param>
        public PluginDiscovery(PluginManager pluginManager)
        {
            _pluginManager = pluginManager ?? throw new ArgumentNullException(nameof(pluginManager));
            _loadedAssemblies = new List<Assembly>();
            _watchers = new Dictionary<string, FileSystemWatcher>();
        }

        /// <summary>
        /// Discovers and registers plugins from the specified assembly.
        /// </summary>
        /// <param name="assembly">The assembly to scan.</param>
        /// <returns>The number of plugins discovered.</returns>
        public int DiscoverPlugins(Assembly assembly)
        {
            if (assembly == null)
            {
                throw new ArgumentNullException(nameof(assembly));
            }

            int count = 0;

            // Discover registry plugins
            var pluginTypes = assembly.GetTypes()
                .Where(t => typeof(IModelRegistryPlugin).IsAssignableFrom(t) &&
                           !t.IsInterface &&
                           !t.IsAbstract)
                .ToList();

            foreach (var pluginType in pluginTypes)
            {
                try
                {
                    var plugin = (IModelRegistryPlugin)Activator.CreateInstance(pluginType);
                    _pluginManager.RegisterPlugin(plugin);
                    count++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PluginDiscovery] Failed to instantiate plugin {pluginType.FullName}: {ex.Message}");
                }
            }

            // Discover extensions
            var extensionTypes = assembly.GetTypes()
                .Where(t => typeof(IModelZooExtension).IsAssignableFrom(t) &&
                           !t.IsInterface &&
                           !t.IsAbstract)
                .ToList();

            foreach (var extensionType in extensionTypes)
            {
                try
                {
                    var extension = (IModelZooExtension)Activator.CreateInstance(extensionType);
                    _pluginManager.RegisterExtension(extension);
                    count++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PluginDiscovery] Failed to instantiate extension {extensionType.FullName}: {ex.Message}");
                }
            }

            lock (_lock)
            {
                if (!_loadedAssemblies.Contains(assembly))
                {
                    _loadedAssemblies.Add(assembly);
                }
            }

            Console.WriteLine($"[PluginDiscovery] Discovered {count} plugins/extensions from assembly: {assembly.GetName().Name}");
            return count;
        }

        /// <summary>
        /// Discovers and registers plugins from all assemblies in the current AppDomain.
        /// </summary>
        /// <returns>The total number of plugins discovered.</returns>
        public int DiscoverAllPlugins()
        {
            int total = 0;
            var assemblies = AppDomain.CurrentDomain.GetAssemblies();

            foreach (var assembly in assemblies)
            {
                try
                {
                    total += DiscoverPlugins(assembly);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PluginDiscovery] Error scanning assembly {assembly.GetName().Name}: {ex.Message}");
                }
            }

            Console.WriteLine($"[PluginDiscovery] Total plugins discovered: {total}");
            return total;
        }

        /// <summary>
        /// Loads plugins from a directory containing plugin DLLs.
        /// </summary>
        /// <param name="directory">The directory to scan for plugin DLLs.</param>
        /// <param name="searchPattern">The file search pattern (default: *.dll).</param>
        /// <returns>The number of plugins loaded.</returns>
        public int LoadPluginsFromDirectory(string directory, string searchPattern = "*.dll")
        {
            if (string.IsNullOrEmpty(directory))
            {
                throw new ArgumentException("Directory cannot be null or empty", nameof(directory));
            }

            if (!Directory.Exists(directory))
            {
                throw new DirectoryNotFoundException($"Directory not found: {directory}");
            }

            int count = 0;
            var dllFiles = Directory.GetFiles(directory, searchPattern, SearchOption.TopDirectoryOnly);

            foreach (var dllFile in dllFiles)
            {
                try
                {
                    count += LoadPluginAssembly(dllFile);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PluginDiscovery] Error loading DLL {dllFile}: {ex.Message}");
                }
            }

            Console.WriteLine($"[PluginDiscovery] Loaded {count} plugins from directory: {directory}");
            return count;
        }

        /// <summary>
        /// Loads a plugin assembly from the specified path.
        /// </summary>
        /// <param name="assemblyPath">The path to the assembly file.</param>
        /// <returns>The number of plugins loaded.</returns>
        public int LoadPluginAssembly(string assemblyPath)
        {
            if (string.IsNullOrEmpty(assemblyPath))
            {
                throw new ArgumentException("Assembly path cannot be null or empty", nameof(assemblyPath));
            }

            if (!File.Exists(assemblyPath))
            {
                throw new FileNotFoundException($"Assembly not found: {assemblyPath}");
            }

            var assembly = Assembly.LoadFrom(assemblyPath);
            return DiscoverPlugins(assembly);
        }

        /// <summary>
        /// Enables hot-reloading of plugins from a directory.
        /// When plugin DLLs are modified, they are automatically reloaded.
        /// </summary>
        /// <param name="directory">The directory to watch for changes.</param>
        public void EnableHotReload(string directory)
        {
            if (string.IsNullOrEmpty(directory))
            {
                throw new ArgumentException("Directory cannot be null or empty", nameof(directory));
            }

            if (!Directory.Exists(directory))
            {
                throw new DirectoryNotFoundException($"Directory not found: {directory}");
            }

            lock (_lock)
            {
                if (_watchers.ContainsKey(directory))
                {
                    Console.WriteLine($"[PluginDiscovery] Hot-reload already enabled for directory: {directory}");
                    return;
                }

                var watcher = new FileSystemWatcher(directory)
                {
                    NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName,
                    Filter = "*.dll",
                    EnableRaisingEvents = true
                };

                watcher.Changed += async (sender, e) =>
                {
                    await Task.Delay(500); // Wait for file write to complete

                    if (e.ChangeType == WatcherChangeTypes.Changed ||
                        e.ChangeType == WatcherChangeTypes.Created)
                    {
                        Console.WriteLine($"[PluginDiscovery] Detected change in plugin: {e.Name}");
                        try
                        {
                            LoadPluginAssembly(e.FullPath);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[PluginDiscovery] Error reloading plugin {e.FullPath}: {ex.Message}");
                        }
                    }
                };

                watcher.Created += async (sender, e) =>
                {
                    await Task.Delay(500);
                    Console.WriteLine($"[PluginDiscovery] Detected new plugin: {e.Name}");
                    try
                    {
                        LoadPluginAssembly(e.FullPath);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[PluginDiscovery] Error loading new plugin {e.FullPath}: {ex.Message}");
                    }
                };

                _watchers[directory] = watcher;
                Console.WriteLine($"[PluginDiscovery] Hot-reload enabled for directory: {directory}");
            }
        }

        /// <summary>
        /// Disables hot-reloading for a directory.
        /// </summary>
        /// <param name="directory">The directory to stop watching.</param>
        public void DisableHotReload(string directory)
        {
            if (string.IsNullOrEmpty(directory))
            {
                throw new ArgumentException("Directory cannot be null or empty", nameof(directory));
            }

            lock (_lock)
            {
                if (_watchers.TryGetValue(directory, out var watcher))
                {
                    watcher.Dispose();
                    _watchers.Remove(directory);
                    Console.WriteLine($"[PluginDiscovery] Hot-reload disabled for directory: {directory}");
                }
            }
        }

        /// <summary>
        /// Disables all hot-reload watchers.
        /// </summary>
        public void DisableAllHotReload()
        {
            lock (_lock)
            {
                foreach (var watcher in _watchers.Values)
                {
                    watcher.Dispose();
                }
                _watchers.Clear();
                Console.WriteLine("[PluginDiscovery] All hot-reload watchers disabled");
            }
        }

        /// <summary>
        /// Gets a list of all loaded assemblies.
        /// </summary>
        /// <returns>A read-only list of loaded assemblies.</returns>
        public IReadOnlyList<Assembly> GetLoadedAssemblies()
        {
            lock (_lock)
            {
                return _loadedAssemblies.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets a list of all directories being watched for hot-reload.
        /// </summary>
        /// <returns>A read-only list of directory paths.</returns>
        public IReadOnlyList<string> GetWatchedDirectories()
        {
            lock (_lock)
            {
                return _watchers.Keys.ToList().AsReadOnly();
            }
        }
    }
}
