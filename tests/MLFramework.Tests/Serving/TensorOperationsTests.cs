using Microsoft.VisualStudio.TestTools.UnitTesting;
using RitterFramework.Core.Tensor;
using MLFramework.Serving;
using System.Collections.Generic;

namespace MLFramework.Tests.Serving;

[TestClass]
public class TensorOperationsTests
{
    [TestMethod]
    public void StackWithPadding_WithSameLengthTensors_NoPadding()
    {
        var tensors = new List<Tensor>
        {
            Tensor.FromArray(new float[] { 1, 2, 3 }),
            Tensor.FromArray(new float[] { 4, 5, 6 }),
            Tensor.FromArray(new float[] { 7, 8, 9 })
        };

        var result = TensorOperations.StackWithPadding(tensors);

        Assert.AreEqual(3, result.StackedTensor.Shape[0]); // Batch size
        Assert.AreEqual(3, result.StackedTensor.Shape[1]); // Sequence length
        Assert.AreEqual(3, result.OriginalLengths.Length);
    }

    [TestMethod]
    public void StackWithPadding_WithVariableLengthTensors_PostPads()
    {
        var tensors = new List<Tensor>
        {
            Tensor.FromArray(new float[] { 1, 2 }),
            Tensor.FromArray(new float[] { 3, 4, 5, 6 }),
            Tensor.FromArray(new float[] { 7, 8, 9 })
        };

        var result = TensorOperations.StackWithPadding(
            tensors,
            paddingValue: -1f,
            strategy: PaddingStrategy.Post);

        Assert.AreEqual(3, result.StackedTensor.Shape[0]); // Batch size
        Assert.AreEqual(4, result.StackedTensor.Shape[1]); // Max sequence length
        CollectionAssert.AreEqual(new[] { 2, 4, 3 }, result.OriginalLengths);

        // Check that padding is -1f
        // First tensor (length 2): should have padding at the end
        Assert.AreEqual(1f, result.StackedTensor[new int[] { 0, 0 }]);
        Assert.AreEqual(2f, result.StackedTensor[new int[] { 0, 1 }]);
        Assert.AreEqual(-1f, result.StackedTensor[new int[] { 0, 2 }]);
        Assert.AreEqual(-1f, result.StackedTensor[new int[] { 0, 3 }]);
    }

    [TestMethod]
    public void StackWithPadding_WithVariableLengthTensors_PrePads()
    {
        var tensors = new List<Tensor>
        {
            Tensor.FromArray(new float[] { 1, 2 }),
            Tensor.FromArray(new float[] { 3, 4, 5, 6 }),
            Tensor.FromArray(new float[] { 7, 8, 9 })
        };

        var result = TensorOperations.StackWithPadding(
            tensors,
            paddingValue: 0f,
            strategy: PaddingStrategy.Pre);

        Assert.AreEqual(3, result.StackedTensor.Shape[0]); // Batch size
        Assert.AreEqual(4, result.StackedTensor.Shape[1]); // Max sequence length
        CollectionAssert.AreEqual(new[] { 2, 4, 3 }, result.OriginalLengths);

        // Check that padding is 0f
        // First tensor (length 2): should have padding at the beginning
        Assert.AreEqual(0f, result.StackedTensor[new int[] { 0, 0 }]);
        Assert.AreEqual(0f, result.StackedTensor[new int[] { 0, 1 }]);
        Assert.AreEqual(1f, result.StackedTensor[new int[] { 0, 2 }]);
        Assert.AreEqual(2f, result.StackedTensor[new int[] { 0, 3 }]);
    }

    [TestMethod]
    public void StackWithPadding_WithEmptyList_Throws()
    {
        var tensors = new List<Tensor>();
        
        Assert.ThrowsException<ArgumentException>(() =>
        {
            TensorOperations.StackWithPadding(tensors);
        });
    }

    [TestMethod]
    public void StackWithPadding_WithNullList_Throws()
    {
        Assert.ThrowsException<ArgumentNullException>(() =>
        {
            TensorOperations.StackWithPadding(null);
        });
    }

    [TestMethod]
    public void StackWithPadding_WithMismatchedRanks_Throws()
    {
        var tensors = new List<Tensor>
        {
            Tensor.FromArray(new float[] { 1, 2, 3 }),
            new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 })
        };
        
        Assert.ThrowsException<ArgumentException>(() =>
        {
            TensorOperations.StackWithPadding(tensors);
        });
    }

    [TestMethod]
    public void Unstack_WithOriginalLengths_ReturnsCorrectSizes()
    {
        var stackedData = new float[15]; // 3 batches x 5 sequence length
        for (int i = 0; i < 15; i++)
        {
            stackedData[i] = i + 1;
        }
        var stacked = new Tensor(stackedData, new int[] { 3, 5 });
        var lengths = new[] { 2, 5, 3 };

        var result = TensorOperations.Unstack(stacked, lengths);

        Assert.AreEqual(3, result.Count);
        Assert.AreEqual(2, result[0].Shape[0]);
        Assert.AreEqual(5, result[1].Shape[0]);
        Assert.AreEqual(3, result[2].Shape[0]);
    }

    [TestMethod]
    public void Unstack_WithNullStackedTensor_Throws()
    {
        Assert.ThrowsException<ArgumentNullException>(() =>
        {
            TensorOperations.Unstack(null, new[] { 1, 2, 3 });
        });
    }

    [TestMethod]
    public void Unstack_WithEmptyOriginalLengths_Throws()
    {
        var stacked = new Tensor(new float[6], new int[] { 3, 2 });
        
        Assert.ThrowsException<ArgumentException>(() =>
        {
            TensorOperations.Unstack(stacked, new int[0]);
        });
    }

    [TestMethod]
    public void Unstack_WithMismatchedBatchSize_Throws()
    {
        var stacked = new Tensor(new float[6], new int[] { 3, 2 });
        var lengths = new[] { 1, 2 }; // Only 2 lengths but batch size is 3
        
        Assert.ThrowsException<ArgumentException>(() =>
        {
            TensorOperations.Unstack(stacked, lengths);
        });
    }

    [TestMethod]
    public void StackAndUnstack_RoundTrip_PreservesData()
    {
        var originalTensors = new List<Tensor>
        {
            Tensor.FromArray(new float[] { 1, 2 }),
            Tensor.FromArray(new float[] { 3, 4, 5 }),
            Tensor.FromArray(new float[] { 6 })
        };

        // Stack
        var stackedResult = TensorOperations.StackWithPadding(originalTensors);

        // Unstack
        var unstackedTensors = TensorOperations.Unstack(stackedResult.StackedTensor, stackedResult.OriginalLengths);

        // Verify data preservation
        Assert.AreEqual(originalTensors.Count, unstackedTensors.Count);
        for (int i = 0; i < originalTensors.Count; i++)
        {
            Assert.AreEqual(originalTensors[i].Shape[0], unstackedTensors[i].Shape[0]);
            for (int j = 0; j < originalTensors[i].Shape[0]; j++)
            {
                Assert.AreEqual(
                    originalTensors[i].Data[j],
                    unstackedTensors[i].Data[j],
                    0.0001f,
                    $"Data mismatch at tensor {i}, position {j}"
                );
            }
        }
    }

    [TestMethod]
    public void StackWithPadding_WithCustomPaddingValue_UsesCorrectValue()
    {
        var tensors = new List<Tensor>
        {
            Tensor.FromArray(new float[] { 1, 2 }),
            Tensor.FromArray(new float[] { 3, 4, 5 })
        };

        var result = TensorOperations.StackWithPadding(tensors, paddingValue: 99.5f);

        // Check padding value
        Assert.AreEqual(99.5f, result.StackedTensor[new int[] { 0, 2 }]);
    }

    [TestMethod]
    public void StackWithPadding_BatchIndices_AreCorrect()
    {
        var tensors = new List<Tensor>
        {
            Tensor.FromArray(new float[] { 1 }),
            Tensor.FromArray(new float[] { 2 }),
            Tensor.FromArray(new float[] { 3 })
        };

        var result = TensorOperations.StackWithPadding(tensors);

        CollectionAssert.AreEqual(new[] { 0, 1, 2 }, result.BatchIndices);
    }
}
