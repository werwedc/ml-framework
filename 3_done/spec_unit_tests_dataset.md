# Spec: Unit Tests - Dataset Interface

## Overview
Comprehensive unit tests for dataset interface and implementations.

## Test Structure
```
tests/
  Data/
    DatasetTests.cs
    ListDatasetTests.cs
    ArrayDatasetTests.cs
    InMemoryDatasetTests.cs
```

## Test Cases

### 1. IDataset Interface Tests

**Test Fixture:**
```csharp
public abstract class DatasetTestBase
{
    protected abstract IDataset<int> CreateDataset(int count);
}
```

**Test Cases:**

**Count Property:**
- `Count_ReturnsCorrectCount` - Verify Count matches dataset size
- `Count_ZeroItems_ReturnsZero` - Handle empty dataset
- `Count_LargeDataset_ReturnsLargeNumber` - Test with 1,000,000 items

**GetItem Method:**
- `GetItem_ValidIndex_ReturnsCorrectItem` - Correct item for valid index
- `GetItem_FirstIndex_ReturnsFirstItem` - Index 0 returns first item
- `GetItem_LastIndex_ReturnsLastItem` - Index Count-1 returns last item
- `GetItem_NegativeIndex_LastItem` - Index -1 returns last item
- `GetItem_NegativeIndex_FirstItem` - Index -Count returns first item
- `GetItem_OutOfRangeLower_ThrowsArgumentOutOfRangeException` - Index < -Count
- `GetItem_OutOfRangeUpper_ThrowsArgumentOutOfRangeException` - Index >= Count
- `GetItem_MultipleCalls_SameItem` - Consistency of repeated calls

### 2. ListDataset Tests

**Constructor Tests:**
- `Constructor_NullList_ThrowsArgumentNullException` - Null validation
- `Constructor_EmptyList_CreatesDataset` - Empty list handled correctly
- `Constructor_NonEmptyList_CreatesDataset` - Normal case
- `Constructor_LargeList_CreatesDataset` - Large list (10,000+ items)

**Count Tests:**
- `Count_ReturnsListCount` - Matches List.Count
- `Count_AfterListModification_ReturnsOriginalCount` - Immutable after construction

**GetItem Tests:**
- `GetItem_ReturnsCorrectItem` - Basic retrieval
- `GetItem_AfterListModification_ReturnsOriginalItems` - Snapshot behavior
- `GetItem_ConcurrentAccess_ThreadSafe` - Multiple threads can read

### 3. ArrayDataset Tests

**Constructor Tests:**
- `Constructor_NullArray_ThrowsArgumentNullException` - Null validation
- `Constructor_EmptyArray_CreatesDataset` - Empty array handled correctly
- `Constructor_NonEmptyArray_CreatesDataset` - Normal case
- `Constructor_LargeArray_CreatesDataset` - Large array (10,000+ items)

**Count Tests:**
- `Count_ReturnsArrayLength` - Matches array.Length
- `Count_AfterArrayModification_ReturnsOriginalCount` - Immutable after construction

**GetItem Tests:**
- `GetItem_ReturnsCorrectItem` - Basic retrieval
- `GetItem_DirectArrayAccess_Performance` - No wrapper overhead
- `GetItem_ConcurrentAccess_ThreadSafe` - Multiple threads can read

### 4. InMemoryDataset Tests

**FromEnumerable Tests:**
- `FromEnumerable_NullEnumerable_ThrowsArgumentNullException` - Null validation
- `FromEnumerable_List_ReturnsListDataset` - List optimization
- `FromEnumerable_Array_ReturnsArrayDataset` - Array optimization
- `FromEnumerable_GenericEnumerable_ReturnsArrayDataset` - Materialization
- `FromEnumerable_LargeEnumerable_ReturnsCorrectDataset` - Large dataset
- `FromEnumerable_EmptyEnumerable_ReturnsEmptyDataset` - Edge case

### 5. Performance Tests (Optional)

**GetItem Performance:**
- `GetItem_ArrayDataset_VsListDataset_Performance` - Array should be faster
- `GetItem_LargeDataset_Performance` - Test with 1,000,000 items
- `GetItem_ConcurrentAccess_Scalability` - Test with 10 concurrent threads

### 6. Edge Cases

**Empty Dataset:**
- `EmptyDataset_CountZero` - Count is 0
- `EmptyDataset_GetItem_Throws` - GetItem throws on empty dataset
- `EmptyDataset_Iteration_NoItems` - Enumerable yields nothing

**Single Item Dataset:**
- `SingleItemDataset_CountOne` - Count is 1
- `SingleItemDataset_GetItemZero_ReturnsItem` - Only valid index is 0

**Null Items:**
- `DatasetWithNullItems_GetItem_ReturnsNull` - Can store null values
- `DatasetWithNullItems_Count_IncludesNulls` - Nulls counted in Count

### 7. Thread Safety Tests

**Concurrent Reads:**
- `ConcurrentGetItem_MultipleThreads_NoErrors` - 10 threads, 1000 reads each
- `ConcurrentGetItem_MultipleThreads_CorrectResults` - Verify all results correct
- `ConcurrentGetItem_HighContention_RaceFree` - Stress test with 100 threads

## Test Utilities

**Helper Methods:**
```csharp
private static int[] CreateTestData(int count)
{
    return Enumerable.Range(0, count).ToArray();
}

private static void AssertDatasetEquals(IDataset<int> dataset, int[] expected)
{
    Assert.AreEqual(expected.Length, dataset.Count);
    for (int i = 0; i < expected.Length; i++)
    {
        Assert.AreEqual(expected[i], dataset.GetItem(i));
    }
}
```

## Success Criteria
- [ ] All constructor validations tested
- [ ] All edge cases covered (empty, single item, null items)
- [ ] Negative indexing tested thoroughly
- [ ] Out of range exceptions tested for all boundaries
- [ ] Thread safety verified with concurrent access tests
- [ ] Performance tests demonstrate array optimization
- [ ] Test coverage > 95%
- [ ] All tests pass consistently

## Notes
- Use xUnit or NUnit testing framework
- Use TestCase attribute for parameterized tests where appropriate
- Performance tests should be marked with explicit attribute
- Thread safety tests should use Task.WhenAll for parallel execution
- Mock any external dependencies
