using System;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Represents an asynchronous pipeline operation
    /// </summary>
    public class AsyncOperation
    {
        /// <summary>
        /// Unique ID for this operation
        /// </summary>
        public Guid Id { get; }

        /// <summary>
        /// Type of operation
        /// </summary>
        public OperationType Type { get; }

        /// <summary>
        /// Micro-batch index
        /// </summary>
        public int MicroBatchIndex { get; }

        /// <summary>
        /// Stream index
        /// </summary>
        public int StreamIndex { get; }

        /// <summary>
        /// Task representing the operation
        /// </summary>
        public System.Threading.Tasks.Task Task { get; }

        /// <summary>
        /// Whether the operation is completed
        /// </summary>
        public bool IsCompleted => Task.IsCompleted;

        /// <summary>
        /// Whether the operation faulted
        /// </summary>
        public bool IsFaulted => Task.IsFaulted;

        /// <summary>
        /// Whether the operation was canceled
        /// </summary>
        public bool IsCanceled => Task.IsCanceled;

        public AsyncOperation(
            Guid id,
            OperationType type,
            int microBatchIndex,
            int streamIndex,
            System.Threading.Tasks.Task task)
        {
            if (task == null)
                throw new ArgumentNullException(nameof(task));

            Id = id;
            Type = type;
            MicroBatchIndex = microBatchIndex;
            StreamIndex = streamIndex;
            Task = task;
        }
    }

    /// <summary>
    /// Type of async operation in pipeline
    /// </summary>
    public enum OperationType
    {
        /// <summary>
        /// Forward computation pass
        /// </summary>
        Forward,

        /// <summary>
        /// Backward computation pass
        /// </summary>
        Backward,

        /// <summary>
        /// Send activation tensor forward
        /// </summary>
        SendForward,

        /// <summary>
        /// Receive activation tensor from previous stage
        /// </summary>
        ReceiveForward,

        /// <summary>
        /// Send gradient tensor backward
        /// </summary>
        SendBackward,

        /// <summary>
        /// Receive gradient tensor from next stage
        /// </summary>
        ReceiveBackward
    }
}
