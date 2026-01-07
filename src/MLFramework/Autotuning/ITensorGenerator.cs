using MLFramework.Core;
using MLFramework.Fusion;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Autotuning;

/// <summary>
/// Interface for generating tensors for benchmarking
/// </summary>
public interface ITensorGenerator
{
    Tensor GenerateRandomTensor(TensorShape shape, DataType dataType);
}
