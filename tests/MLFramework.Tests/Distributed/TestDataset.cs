using MLFramework.Data;
using MLFramework.Tensor;
using System;

namespace MLFramework.Tests.Distributed
{
    /// <summary>
    /// Simple dataset for testing.
    /// </summary>
    public class TestDataset : Dataset
    {
        private readonly int _size;
        private readonly Func<int, Tensor> _dataGenerator;

        public TestDataset(int size, Func<int, Tensor> dataGenerator = null)
        {
            _size = size;
            _dataGenerator = dataGenerator ?? (i => Tensor.Random(new long[] { 10, 10 }));
        }

        public override int Count => _size;

        public override Tensor GetItem(int index)
        {
            if (index < 0 || index >= _size)
            {
                throw new IndexOutOfRangeException($"Index {index} out of range for dataset of size {_size}");
            }
            return _dataGenerator(index);
        }
    }

    /// <summary>
    /// Simple model for testing DDP.
    /// </summary>
    public class SimpleModel
    {
        public Tensor Weight { get; private set; }
        public Tensor Bias { get; private set; }

        public SimpleModel()
        {
            Weight = Tensor.Random(new long[] { 10, 10 });
            Bias = Tensor.Random(new long[] { 10 });
        }

        public Tensor Forward(Tensor input)
        {
            var output = Tensor.MatMul(input, Weight);
            output = output.Add(Bias);
            return output;
        }

        public Tensor[] GetParameters()
        {
            return new[] { Weight, Bias };
        }
    }
}
