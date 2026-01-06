using MLFramework.Core;
using RitterFramework.Core.Tensor;
using System;
using System.Runtime.InteropServices;

namespace MLFramework.Distributed.Gloo
{
    /// <summary>
    /// Implements AllReduce using Gloo's ring algorithm.
    /// </summary>
    internal class GlooAllReduce
    {
        private readonly IntPtr _context;
        private readonly int _rank;
        private readonly string _backendName;

        public GlooAllReduce(IntPtr context, int rank, string backendName)
        {
            _context = context;
            _rank = rank;
            _backendName = backendName;
        }

        /// <summary>
        /// Perform AllReduce on a CPU tensor using Gloo's ring algorithm.
        /// </summary>
        public void AllReduceCPU(Tensor tensor, ReduceOp op)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            // Convert from RitterFramework.Core.DataType to MLFramework.Core.DataType
            MLFramework.Core.DataType mlDtype = tensor.Dtype switch
            {
                RitterFramework.Core.DataType.Float32 => MLFramework.Core.DataType.Float32,
                RitterFramework.Core.DataType.Float64 => MLFramework.Core.DataType.Float64,
                RitterFramework.Core.DataType.Int32 => MLFramework.Core.DataType.Int32,
                RitterFramework.Core.DataType.Int64 => MLFramework.Core.DataType.Int64,
                RitterFramework.Core.DataType.Int16 => MLFramework.Core.DataType.Int16,
                RitterFramework.Core.DataType.Int8 => MLFramework.Core.DataType.Int8,
                RitterFramework.Core.DataType.Bool => MLFramework.Core.DataType.Bool,
                _ => throw new ArgumentException($"Unsupported data type: {tensor.Dtype}", nameof(tensor))
            };

            var dataType = GetGlooDataType(mlDtype);
            var opType = GetGlooOp(op);

            // Pin the data array to prevent GC from moving it
            GCHandle handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                long count = tensor.NumElements();

                GlooNative.gloo_allreduce(_context, ptr, ptr, count, dataType, opType);
            }
            catch (DllNotFoundException ex)
            {
                throw new CommunicationException("Gloo library not found", _rank, _backendName, ex);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"AllReduce failed: {ex.Message}", _rank, _backendName, ex);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Perform AllReduce on a CUDA tensor.
        /// Gloo transfers to CPU, reduces, and transfers back.
        /// </summary>
        public void AllReduceCUDA(Tensor tensor, ReduceOp op)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            var device = tensor.GetDevice();

            if (device.Type != DeviceType.CUDA)
            {
                throw new ArgumentException("Tensor must be on CUDA device", nameof(tensor));
            }

            try
            {
                // Copy to CPU
                var cpuTensor = tensor.To(Device.CreateCpu());

                // Reduce on CPU
                AllReduceCPU(cpuTensor, op);

                // Copy back to CUDA
                // For now, just update the original tensor data
                // In a real implementation, this would use CUDA memory operations
                tensor.Copy_(cpuTensor);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"CUDA AllReduce failed: {ex.Message}", _rank, _backendName, ex);
            }
        }

        /// <summary>
        /// Perform Broadcast on a CPU tensor.
        /// </summary>
        public void BroadcastCPU(Tensor tensor, int root)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (root < 0 || root >= int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(root), "Root rank must be non-negative");
            }

            // Convert from RitterFramework.Core.DataType to MLFramework.Core.DataType
            MLFramework.Core.DataType mlDtype = tensor.Dtype switch
            {
                RitterFramework.Core.DataType.Float32 => MLFramework.Core.DataType.Float32,
                RitterFramework.Core.DataType.Float64 => MLFramework.Core.DataType.Float64,
                RitterFramework.Core.DataType.Int32 => MLFramework.Core.DataType.Int32,
                RitterFramework.Core.DataType.Int64 => MLFramework.Core.DataType.Int64,
                RitterFramework.Core.DataType.Int16 => MLFramework.Core.DataType.Int16,
                RitterFramework.Core.DataType.Int8 => MLFramework.Core.DataType.Int8,
                RitterFramework.Core.DataType.Bool => MLFramework.Core.DataType.Bool,
                _ => throw new ArgumentException($"Unsupported data type: {tensor.Dtype}", nameof(tensor))
            };

            var dataType = GetGlooDataType(mlDtype);

            // Pin the data array to prevent GC from moving it
            GCHandle handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                long count = tensor.NumElements();

                GlooNative.gloo_broadcast(_context, ptr, count, dataType, root);
            }
            catch (DllNotFoundException ex)
            {
                throw new CommunicationException("Gloo library not found", _rank, _backendName, ex);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"Broadcast failed: {ex.Message}", _rank, _backendName, ex);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Perform Broadcast on a CUDA tensor.
        /// </summary>
        public void BroadcastCUDA(Tensor tensor, int root)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            var device = tensor.GetDevice();

            if (device.Type != DeviceType.CUDA)
            {
                throw new ArgumentException("Tensor must be on CUDA device", nameof(tensor));
            }

            try
            {
                // Copy to CPU
                var cpuTensor = tensor.To(Device.CreateCpu());

                // Broadcast on CPU
                BroadcastCPU(cpuTensor, root);

                // Copy back to CUDA
                tensor.Copy_(cpuTensor);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"CUDA Broadcast failed: {ex.Message}", _rank, _backendName, ex);
            }
        }

        /// <summary>
        /// Perform Send on a CPU tensor.
        /// </summary>
        public void SendCPU(Tensor tensor, int dst)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (dst < 0 || dst >= int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(dst), "Destination rank must be non-negative");
            }

            // Convert from RitterFramework.Core.DataType to MLFramework.Core.DataType
            MLFramework.Core.DataType mlDtype = tensor.Dtype switch
            {
                RitterFramework.Core.DataType.Float32 => MLFramework.Core.DataType.Float32,
                RitterFramework.Core.DataType.Float64 => MLFramework.Core.DataType.Float64,
                RitterFramework.Core.DataType.Int32 => MLFramework.Core.DataType.Int32,
                RitterFramework.Core.DataType.Int64 => MLFramework.Core.DataType.Int64,
                RitterFramework.Core.DataType.Int16 => MLFramework.Core.DataType.Int16,
                RitterFramework.Core.DataType.Int8 => MLFramework.Core.DataType.Int8,
                RitterFramework.Core.DataType.Bool => MLFramework.Core.DataType.Bool,
                _ => throw new ArgumentException($"Unsupported data type: {tensor.Dtype}", nameof(tensor))
            };

            var dataType = GetGlooDataType(mlDtype);

            // Pin the data array to prevent GC from moving it
            GCHandle handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                long count = tensor.NumElements();

                GlooNative.gloo_send(_context, ptr, count, dataType, dst);
            }
            catch (DllNotFoundException ex)
            {
                throw new CommunicationException("Gloo library not found", _rank, _backendName, ex);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"Send failed: {ex.Message}", _rank, _backendName, ex);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Perform Send on a CUDA tensor.
        /// </summary>
        public void SendCUDA(Tensor tensor, int dst)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            var device = tensor.GetDevice();

            if (device.Type != DeviceType.CUDA)
            {
                throw new ArgumentException("Tensor must be on CUDA device", nameof(tensor));
            }

            try
            {
                // Copy to CPU
                var cpuTensor = tensor.To(Device.CreateCpu());

                // Send on CPU
                SendCPU(cpuTensor, dst);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"CUDA Send failed: {ex.Message}", _rank, _backendName, ex);
            }
        }

        /// <summary>
        /// Perform Recv on a CPU tensor.
        /// </summary>
        public void RecvCPU(Tensor tensor, int src)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (src < 0 || src >= int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(src), "Source rank must be non-negative");
            }

            // Convert from RitterFramework.Core.DataType to MLFramework.Core.DataType
            MLFramework.Core.DataType mlDtype = tensor.Dtype switch
            {
                RitterFramework.Core.DataType.Float32 => MLFramework.Core.DataType.Float32,
                RitterFramework.Core.DataType.Float64 => MLFramework.Core.DataType.Float64,
                RitterFramework.Core.DataType.Int32 => MLFramework.Core.DataType.Int32,
                RitterFramework.Core.DataType.Int64 => MLFramework.Core.DataType.Int64,
                RitterFramework.Core.DataType.Int16 => MLFramework.Core.DataType.Int16,
                RitterFramework.Core.DataType.Int8 => MLFramework.Core.DataType.Int8,
                RitterFramework.Core.DataType.Bool => MLFramework.Core.DataType.Bool,
                _ => throw new ArgumentException($"Unsupported data type: {tensor.Dtype}", nameof(tensor))
            };

            var dataType = GetGlooDataType(mlDtype);

            // Pin the data array to prevent GC from moving it
            GCHandle handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                long count = tensor.NumElements();

                GlooNative.gloo_recv(_context, ptr, count, dataType, src);
            }
            catch (DllNotFoundException ex)
            {
                throw new CommunicationException("Gloo library not found", _rank, _backendName, ex);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"Recv failed: {ex.Message}", _rank, _backendName, ex);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Perform Recv on a CUDA tensor.
        /// </summary>
        public void RecvCUDA(Tensor tensor, int src)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            var device = tensor.GetDevice();

            if (device.Type != DeviceType.CUDA)
            {
                throw new ArgumentException("Tensor must be on CUDA device", nameof(tensor));
            }

            try
            {
                // Copy to CPU
                var cpuTensor = tensor.To(Device.CreateCpu());

                // Recv on CPU
                RecvCPU(cpuTensor, src);

                // Copy back to CUDA
                tensor.Copy_(cpuTensor);
            }
            catch (Exception ex)
            {
                throw new CommunicationException($"CUDA Recv failed: {ex.Message}", _rank, _backendName, ex);
            }
        }

        /// <summary>
        /// Converts DataType to Gloo's data type enum.
        /// </summary>
        private glooDataType_t GetGlooDataType(DataType dtype)
        {
            return dtype switch
            {
                DataType.Int8 => glooDataType_t.glooInt8,
                DataType.UInt8 => glooDataType_t.glooUint8,
                DataType.Int16 => glooDataType_t.glooInt16,
                DataType.Int32 => glooDataType_t.glooInt32,
                DataType.Int64 => glooDataType_t.glooInt64,
                DataType.Float16 => glooDataType_t.glooFloat16,
                DataType.Float32 => glooDataType_t.glooFloat32,
                DataType.Float64 => glooDataType_t.glooFloat64,
                DataType.BFloat16 => glooDataType_t.glooFloat16, // Gloo doesn't have BF16, use FP16
                DataType.Bool => glooDataType_t.glooUint8,
                _ => throw new ArgumentException($"Unsupported data type: {dtype}", nameof(dtype))
            };
        }

        /// <summary>
        /// Converts ReduceOp to Gloo's reduction operation enum.
        /// </summary>
        private glooRedOp_t GetGlooOp(ReduceOp op)
        {
            return op switch
            {
                ReduceOp.Sum => glooRedOp_t.glooSum,
                ReduceOp.Product => glooRedOp_t.glooProduct,
                ReduceOp.Max => glooRedOp_t.glooMax,
                ReduceOp.Min => glooRedOp_t.glooMin,
                ReduceOp.Avg => glooRedOp_t.glooAvg,
                _ => throw new ArgumentException($"Unsupported reduction operation: {op}", nameof(op))
            };
        }
    }
}
