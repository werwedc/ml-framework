using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Registry for custom autograd functions, allowing registration and retrieval by name.
/// </summary>
public static class FunctionRegistry
{
    private static readonly Dictionary<string, Type> _registry;
    private static readonly object _lock = new object();

    static FunctionRegistry()
    {
        _registry = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Registers a custom autograd function with the given name.
    /// </summary>
    /// <typeparam name="T">The type of the autograd function. Must inherit from AutogradFunction.</typeparam>
    /// <param name="name">The name to register the function under.</param>
    /// <exception cref="ArgumentNullException">Thrown when name is null or empty.</exception>
    /// <exception cref="ArgumentException">Thrown when T does not inherit from AutogradFunction or is not a class.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a function with the same name is already registered.</exception>
    public static void Register<T>(string name) where T : AutogradFunction, new()
    {
        if (string.IsNullOrEmpty(name))
            throw new ArgumentNullException(nameof(name));

        if (!typeof(T).IsClass)
            throw new ArgumentException($"Type {typeof(T).Name} must be a class");

        if (!typeof(AutogradFunction).IsAssignableFrom(typeof(T)))
            throw new ArgumentException($"Type {typeof(T).Name} must inherit from AutogradFunction");

        lock (_lock)
        {
            if (_registry.ContainsKey(name))
                throw new InvalidOperationException($"A function with name '{name}' is already registered");

            _registry[name] = typeof(T);
        }
    }

    /// <summary>
    /// Retrieves the type of a registered function by name.
    /// </summary>
    /// <param name="name">The name of the function.</param>
    /// <returns>The type of the registered function.</returns>
    /// <exception cref="ArgumentNullException">Thrown when name is null or empty.</exception>
    /// <exception cref="KeyNotFoundException">Thrown when no function with the given name is registered.</exception>
    public static Type GetFunctionType(string name)
    {
        if (string.IsNullOrEmpty(name))
            throw new ArgumentNullException(nameof(name));

        lock (_lock)
        {
            if (!_registry.TryGetValue(name, out var functionType))
                throw new KeyNotFoundException($"No function registered with name '{name}'");

            return functionType;
        }
    }

    /// <summary>
    /// Checks if a function with the given name is registered.
    /// </summary>
    /// <param name="name">The name to check.</param>
    /// <returns>True if a function with the name is registered, false otherwise.</returns>
    public static bool IsRegistered(string name)
    {
        if (string.IsNullOrEmpty(name))
            return false;

        lock (_lock)
        {
            return _registry.ContainsKey(name);
        }
    }

    /// <summary>
    /// Unregisters a function with the given name.
    /// </summary>
    /// <param name="name">The name of the function to unregister.</param>
    /// <returns>True if the function was unregistered, false if it wasn't found.</returns>
    public static bool Unregister(string name)
    {
        if (string.IsNullOrEmpty(name))
            return false;

        lock (_lock)
        {
            return _registry.Remove(name);
        }
    }

    /// <summary>
    /// Creates an instance of a registered function by name.
    /// </summary>
    /// <param name="name">The name of the function to create.</param>
    /// <returns>A new instance of the registered function.</returns>
    /// <exception cref="ArgumentNullException">Thrown when name is null or empty.</exception>
    /// <exception cref="KeyNotFoundException">Thrown when no function with the given name is registered.</exception>
    /// <exception cref="MissingMethodException">Thrown when the function type doesn't have a parameterless constructor.</exception>
    public static AutogradFunction CreateInstance(string name)
    {
        var functionType = GetFunctionType(name);

        if (typeof(AutogradFunction).IsAssignableFrom(functionType))
        {
            // Try to create an instance using reflection
            try
            {
                return (AutogradFunction)Activator.CreateInstance(functionType)!;
            }
            catch (Exception ex)
            {
                throw new MissingMethodException(
                    $"Failed to create instance of {functionType.Name}. Ensure it has a parameterless constructor.",
                    ex);
            }
        }

        throw new InvalidOperationException($"Type {functionType.Name} is not a valid AutogradFunction");
    }

    /// <summary>
    /// Gets all registered function names.
    /// </summary>
    /// <returns>A collection of all registered function names.</returns>
    public static IEnumerable<string> GetAllRegisteredNames()
    {
        lock (_lock)
        {
            return _registry.Keys.ToList();
        }
    }

    /// <summary>
    /// Clears all registered functions.
    /// </summary>
    public static void ClearAll()
    {
        lock (_lock)
        {
            _registry.Clear();
        }
    }
}

/// <summary>
/// Extension methods for applying registered autograd functions to tensors.
/// </summary>
public static class TensorFunctionExtensions
{
    /// <summary>
    /// Applies a registered autograd function to this tensor.
    /// </summary>
    /// <typeparam name="T">The type of the autograd function.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="args">Additional arguments (not used in basic implementation).</param>
    /// <returns>The output tensor.</returns>
    public static Tensor ApplyFunction<T>(this Tensor tensor, params object[] args)
        where T : AutogradFunction, new()
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var function = new T();
        return function.Apply(tensor);
    }

    /// <summary>
    /// Applies a registered autograd function by name to this tensor.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="functionName">The name of the registered function.</param>
    /// <param name="args">Additional arguments (not used in basic implementation).</param>
    /// <returns>The output tensor.</returns>
    public static Tensor ApplyFunction(this Tensor tensor, string functionName, params object[] args)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var function = FunctionRegistry.CreateInstance(functionName);
        return function.Apply(tensor);
    }
}
