namespace RitterFramework.Core.Tensor;

/// <summary>
/// A named tensor wrapper that associates a name with a tensor.
/// Used for parameter tracking in distributed training.
/// </summary>
public class NamedTensor
{
    /// <summary>Name of the tensor/parameter</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>The tensor data</summary>
    public Tensor Tensor { get; set; } = null!;

    /// <summary>
    /// Create a new named tensor.
    /// </summary>
    public NamedTensor()
    {
    }

    /// <summary>
    /// Create a new named tensor with name and tensor.
    /// </summary>
    /// <param name="name">Name of the tensor</param>
    /// <param name="tensor">The tensor</param>
    public NamedTensor(string name, Tensor tensor)
    {
        Name = name;
        Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }
}
