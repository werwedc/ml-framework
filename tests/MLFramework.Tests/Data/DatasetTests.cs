using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data
{
    /// <summary>
    /// Unit tests for the IDataset interface and its implementations.
    /// </summary>
    public class DatasetTests
    {
        #region ListDataset Tests

        [Fact]
        public void ListDataset_Constructor_WithValidList_Succeeds()
        {
            // Arrange
            var items = new List<int> { 1, 2, 3, 4, 5 };

            // Act
            var dataset = new ListDataset<int>(items);

            // Assert
            Assert.NotNull(dataset);
            Assert.Equal(5, dataset.Count);
        }

        [Fact]
        public void ListDataset_Constructor_WithNull_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() => new ListDataset<int>(null));
        }

        [Fact]
        public void ListDataset_GetItem_ValidIndex_ReturnsCorrectItem()
        {
            // Arrange
            var items = new List<int> { 10, 20, 30, 40, 50 };
            var dataset = new ListDataset<int>(items);

            // Act & Assert
            Assert.Equal(10, dataset.GetItem(0));
            Assert.Equal(20, dataset.GetItem(1));
            Assert.Equal(30, dataset.GetItem(2));
            Assert.Equal(40, dataset.GetItem(3));
            Assert.Equal(50, dataset.GetItem(4));
        }

        [Fact]
        public void ListDataset_GetItem_NegativeIndex_ReturnsCorrectItem()
        {
            // Arrange
            var items = new List<int> { 10, 20, 30, 40, 50 };
            var dataset = new ListDataset<int>(items);

            // Act & Assert
            Assert.Equal(50, dataset.GetItem(-1));
            Assert.Equal(40, dataset.GetItem(-2));
            Assert.Equal(30, dataset.GetItem(-3));
            Assert.Equal(20, dataset.GetItem(-4));
            Assert.Equal(10, dataset.GetItem(-5));
        }

        [Fact]
        public void ListDataset_GetItem_InvalidPositiveIndex_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var items = new List<int> { 1, 2, 3 };
            var dataset = new ListDataset<int>(items);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(3));
            Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(100));
        }

        [Fact]
        public void ListDataset_GetItem_InvalidNegativeIndex_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var items = new List<int> { 1, 2, 3 };
            var dataset = new ListDataset<int>(items);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(-4));
            Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(-100));
        }

        [Fact]
        public void ListDataset_GetBatch_ValidIndices_ReturnsCorrectItems()
        {
            // Arrange
            var items = new List<int> { 10, 20, 30, 40, 50 };
            var dataset = new ListDataset<int>(items);
            var indices = new[] { 0, 2, 4 };

            // Act
            var result = dataset.GetBatch(indices);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(10, result[0]);
            Assert.Equal(30, result[1]);
            Assert.Equal(50, result[2]);
        }

        [Fact]
        public void ListDataset_GetBatch_NegativeIndices_ReturnsCorrectItems()
        {
            // Arrange
            var items = new List<int> { 10, 20, 30, 40, 50 };
            var dataset = new ListDataset<int>(items);
            var indices = new[] { -1, -3, -5 };

            // Act
            var result = dataset.GetBatch(indices);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(50, result[0]);
            Assert.Equal(30, result[1]);
            Assert.Equal(10, result[2]);
        }

        [Fact]
        public void ListDataset_GetBatch_NullIndices_ThrowsArgumentNullException()
        {
            // Arrange
            var items = new List<int> { 1, 2, 3 };
            var dataset = new ListDataset<int>(items);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => dataset.GetBatch(null));
        }

        [Fact]
        public void ListDataset_GetBatch_EmptyIndices_ReturnsEmptyArray()
        {
            // Arrange
            var items = new List<int> { 1, 2, 3 };
            var dataset = new ListDataset<int>(items);

            // Act
            var result = dataset.GetBatch(Array.Empty<int>());

            // Assert
            Assert.Empty(result);
        }

        [Fact]
        public void ListDataset_ConcurrentReads_ThreadSafe()
        {
            // Arrange
            var items = new List<int>(Enumerable.Range(0, 1000));
            var dataset = new ListDataset<int>(items);
            var exceptions = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
            const int threadCount = 10;
            const int iterationsPerThread = 100;

            // Act
            var tasks = Enumerable.Range(0, threadCount).Select(threadId =>
                Task.Run(() =>
                {
                    try
                    {
                        for (int i = 0; i < iterationsPerThread; i++)
                        {
                            // Read from various indices including negative
                            int index = i % 1000;
                            int _ = dataset.GetItem(index);
                            int _2 = dataset.GetItem(-index - 1);
                            var batch = dataset.GetBatch(new[] { index, -index - 1 });
                        }
                    }
                    catch (Exception ex)
                    {
                        exceptions.Enqueue(ex);
                    }
                })
            );

            Task.WaitAll(tasks.ToArray());

            // Assert
            Assert.Empty(exceptions);
        }

        #endregion

        #region ArrayDataset Tests

        [Fact]
        public void ArrayDataset_Constructor_WithValidArray_Succeeds()
        {
            // Arrange
            var items = new[] { 1, 2, 3, 4, 5 };

            // Act
            var dataset = new ArrayDataset<int>(items);

            // Assert
            Assert.NotNull(dataset);
            Assert.Equal(5, dataset.Count);
        }

        [Fact]
        public void ArrayDataset_Constructor_WithNull_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() => new ArrayDataset<int>(null));
        }

        [Fact]
        public void ArrayDataset_GetItem_ValidIndex_ReturnsCorrectItem()
        {
            // Arrange
            var items = new[] { 10, 20, 30, 40, 50 };
            var dataset = new ArrayDataset<int>(items);

            // Act & Assert
            Assert.Equal(10, dataset.GetItem(0));
            Assert.Equal(20, dataset.GetItem(1));
            Assert.Equal(30, dataset.GetItem(2));
            Assert.Equal(40, dataset.GetItem(3));
            Assert.Equal(50, dataset.GetItem(4));
        }

        [Fact]
        public void ArrayDataset_GetItem_NegativeIndex_ReturnsCorrectItem()
        {
            // Arrange
            var items = new[] { 10, 20, 30, 40, 50 };
            var dataset = new ArrayDataset<int>(items);

            // Act & Assert
            Assert.Equal(50, dataset.GetItem(-1));
            Assert.Equal(40, dataset.GetItem(-2));
            Assert.Equal(30, dataset.GetItem(-3));
            Assert.Equal(20, dataset.GetItem(-4));
            Assert.Equal(10, dataset.GetItem(-5));
        }

        [Fact]
        public void ArrayDataset_GetItem_InvalidPositiveIndex_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var items = new[] { 1, 2, 3 };
            var dataset = new ArrayDataset<int>(items);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(3));
            Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(100));
        }

        [Fact]
        public void ArrayDataset_GetItem_InvalidNegativeIndex_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var items = new[] { 1, 2, 3 };
            var dataset = new ArrayDataset<int>(items);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(-4));
            Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(-100));
        }

        [Fact]
        public void ArrayDataset_GetBatch_ValidIndices_ReturnsCorrectItems()
        {
            // Arrange
            var items = new[] { 10, 20, 30, 40, 50 };
            var dataset = new ArrayDataset<int>(items);
            var indices = new[] { 0, 2, 4 };

            // Act
            var result = dataset.GetBatch(indices);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(10, result[0]);
            Assert.Equal(30, result[1]);
            Assert.Equal(50, result[2]);
        }

        [Fact]
        public void ArrayDataset_GetBatch_NegativeIndices_ReturnsCorrectItems()
        {
            // Arrange
            var items = new[] { 10, 20, 30, 40, 50 };
            var dataset = new ArrayDataset<int>(items);
            var indices = new[] { -1, -3, -5 };

            // Act
            var result = dataset.GetBatch(indices);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(50, result[0]);
            Assert.Equal(30, result[1]);
            Assert.Equal(10, result[2]);
        }

        [Fact]
        public void ArrayDataset_GetBatch_NullIndices_ThrowsArgumentNullException()
        {
            // Arrange
            var items = new[] { 1, 2, 3 };
            var dataset = new ArrayDataset<int>(items);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => dataset.GetBatch(null));
        }

        [Fact]
        public void ArrayDataset_GetBatch_EmptyIndices_ReturnsEmptyArray()
        {
            // Arrange
            var items = new[] { 1, 2, 3 };
            var dataset = new ArrayDataset<int>(items);

            // Act
            var result = dataset.GetBatch(Array.Empty<int>());

            // Assert
            Assert.Empty(result);
        }

        [Fact]
        public void ArrayDataset_ConcurrentReads_ThreadSafe()
        {
            // Arrange
            var items = Enumerable.Range(0, 1000).ToArray();
            var dataset = new ArrayDataset<int>(items);
            var exceptions = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
            const int threadCount = 10;
            const int iterationsPerThread = 100;

            // Act
            var tasks = Enumerable.Range(0, threadCount).Select(threadId =>
                Task.Run(() =>
                {
                    try
                    {
                        for (int i = 0; i < iterationsPerThread; i++)
                        {
                            // Read from various indices including negative
                            int index = i % 1000;
                            int _ = dataset.GetItem(index);
                            int _2 = dataset.GetItem(-index - 1);
                            var batch = dataset.GetBatch(new[] { index, -index - 1 });
                        }
                    }
                    catch (Exception ex)
                    {
                        exceptions.Enqueue(ex);
                    }
                })
            );

            Task.WaitAll(tasks.ToArray());

            // Assert
            Assert.Empty(exceptions);
        }

        #endregion

        #region InMemoryDataset Tests

        [Fact]
        public void InMemoryDataset_FromEnumerable_WithArray_ReturnsArrayDataset()
        {
            // Arrange
            var items = new[] { 1, 2, 3, 4, 5 };

            // Act
            var dataset = InMemoryDataset.FromEnumerable(items);

            // Assert
            Assert.IsType<ArrayDataset<int>>(dataset);
            Assert.Equal(5, dataset.Count);
            Assert.Equal(1, dataset.GetItem(0));
        }

        [Fact]
        public void InMemoryDataset_FromEnumerable_WithList_ReturnsListDataset()
        {
            // Arrange
            var items = new List<int> { 1, 2, 3, 4, 5 };

            // Act
            var dataset = InMemoryDataset.FromEnumerable(items);

            // Assert
            Assert.IsType<ListDataset<int>>(dataset);
            Assert.Equal(5, dataset.Count);
            Assert.Equal(1, dataset.GetItem(0));
        }

        [Fact]
        public void InMemoryDataset_FromEnumerable_WithIEnumerable_ReturnsArrayDataset()
        {
            // Arrange
            var items = Enumerable.Range(1, 5).Select(x => x);

            // Act
            var dataset = InMemoryDataset.FromEnumerable(items);

            // Assert
            Assert.IsType<ArrayDataset<int>>(dataset);
            Assert.Equal(5, dataset.Count);
            Assert.Equal(1, dataset.GetItem(0));
        }

        [Fact]
        public void InMemoryDataset_FromEnumerable_WithNull_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() => InMemoryDataset.FromEnumerable<int>(null));
        }

        [Fact]
        public void InMemoryDataset_FromList_ReturnsListDataset()
        {
            // Arrange
            var items = new List<int> { 1, 2, 3 };

            // Act
            var dataset = InMemoryDataset.FromList(items);

            // Assert
            Assert.IsType<ListDataset<int>>(dataset);
            Assert.Equal(3, dataset.Count);
        }

        [Fact]
        public void InMemoryDataset_FromList_WithNull_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() => InMemoryDataset.FromList<int>(null));
        }

        [Fact]
        public void InMemoryDataset_FromArray_ReturnsArrayDataset()
        {
            // Arrange
            var items = new[] { 1, 2, 3 };

            // Act
            var dataset = InMemoryDataset.FromArray(items);

            // Assert
            Assert.IsType<ArrayDataset<int>>(dataset);
            Assert.Equal(3, dataset.Count);
        }

        [Fact]
        public void InMemoryDataset_FromArray_WithNull_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() => InMemoryDataset.FromArray<int>(null));
        }

        #endregion

        #region Dataset Base Class Tests

        [Fact]
        public void Dataset_GetBatch_DefaultImplementation_WorksCorrectly()
        {
            // Arrange
            var items = new[] { 1, 2, 3, 4, 5 };
            var dataset = new TestDataset(items);
            var indices = new[] { 0, 2, 4 };

            // Act
            var result = dataset.GetBatch(indices);

            // Assert
            Assert.Equal(3, result.Length);
            Assert.Equal(1, result[0]);
            Assert.Equal(3, result[1]);
            Assert.Equal(5, result[2]);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Dataset_EmptyDataset_CountReturnsZero()
        {
            // Arrange
            var emptyArray = new int[0];
            var arrayDataset = new ArrayDataset<int>(emptyArray);
            var emptyList = new List<int>();
            var listDataset = new ListDataset<int>(emptyList);

            // Act & Assert
            Assert.Equal(0, arrayDataset.Count);
            Assert.Equal(0, listDataset.Count);
        }

        [Fact]
        public void Dataset_EmptyDataset_GetItem_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var emptyArray = new int[0];
            var arrayDataset = new ArrayDataset<int>(emptyArray);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => arrayDataset.GetItem(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => arrayDataset.GetItem(-1));
        }

        [Fact]
        public void Dataset_SingleItemDataset_WorksCorrectly()
        {
            // Arrange
            var items = new[] { 42 };
            var dataset = new ArrayDataset<int>(items);

            // Act & Assert
            Assert.Equal(1, dataset.Count);
            Assert.Equal(42, dataset.GetItem(0));
            Assert.Equal(42, dataset.GetItem(-1));
        }

        [Fact]
        public void Dataset_WithReferenceTypes_WorksCorrectly()
        {
            // Arrange
            var items = new[] { "one", "two", "three" };
            var dataset = new ArrayDataset<string>(items);

            // Act & Assert
            Assert.Equal(3, dataset.Count);
            Assert.Equal("one", dataset.GetItem(0));
            Assert.Equal("three", dataset.GetItem(-1));
        }

        #endregion

        #region Test Helper Classes

        /// <summary>
        /// Test implementation of Dataset{T} to test the base class functionality.
        /// </summary>
        private class TestDataset : Dataset<int>
        {
            private readonly int[] _items;

            public TestDataset(int[] items)
            {
                _items = items;
            }

            public override int Count => _items.Length;

            public override int GetItem(int index)
            {
                int normalizedIndex = NormalizeIndex(index);
                return _items[normalizedIndex];
            }
        }

        #endregion
    }
}
