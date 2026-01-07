namespace RitterFramework.Core.LoRA
{
    /// <summary>
    /// Linear layer with LoRA adaptation.
    /// Wraps a base Linear layer and adds low-rank adaptation matrices.
    /// </summary>
    public class LoraLinear : IModule
    {
        /// <summary>Base linear layer (wrapped)</summary>
        public IModule BaseLayer { get; }

        /// <summary>LoRA rank</summary>
        public int Rank { get; }

        /// <summary>LoRA alpha (scaling factor)</summary>
        public int Alpha { get; }

        /// <summary>Dropout probability</summary>
        public float Dropout { get; }

        /// <summary>LoRA matrix A (down-projection)</summary>
        public Core.Tensor.Tensor LoraA { get; }

        /// <summary>LoRA matrix B (up-projection)</summary>
        public Core.Tensor.Tensor LoraB { get; }

        /// <summary>Scaling factor (alpha / rank)</summary>
        public float Scaling => (float)Alpha / Rank;

        /// <summary>Module name</summary>
        public string Name { get; set; }

        /// <summary>
        /// Constructor.
        /// </summary>
        public LoraLinear(IModule baseLayer, int rank, int alpha, float dropout = 0.0f)
        {
            BaseLayer = baseLayer ?? throw new ArgumentNullException(nameof(baseLayer));
            Rank = rank;
            Alpha = alpha;
            Dropout = dropout;
            Name = baseLayer.Name;

            // Initialize LoRA matrices with random normal distribution
            // These would typically be initialized based on the base layer dimensions
            LoraA = Core.Tensor.Tensor.Zeros(new[] { rank, 1 }); // Placeholder shape
            LoraB = Core.Tensor.Tensor.Zeros(new[] { 1, 1 });    // Placeholder shape

            LoraA.RequiresGrad = true;
            LoraB.RequiresGrad = true;
        }

        /// <summary>
        /// Get trainable parameters (only LoRA matrices).
        /// </summary>
        public IEnumerable<Core.Tensor.Tensor> TrainableParameters()
        {
            yield return LoraA;
            yield return LoraB;
        }

        /// <summary>
        /// Reset LoRA matrices to zero.
        /// </summary>
        public void ResetLoRA()
        {
            var zerosA = new float[LoraA.Size];
            Array.Copy(zerosA, LoraA.Data, zerosA.Length);

            var zerosB = new float[LoraB.Size];
            Array.Copy(zerosB, LoraB.Data, zerosB.Length);
        }

        /// <summary>
        /// Add LoRA weights from an adapter.
        /// </summary>
        public void AddLoRAWeights(Core.Tensor.Tensor loraA, Core.Tensor.Tensor loraB)
        {
            if (loraA == null) throw new ArgumentNullException(nameof(loraA));
            if (loraB == null) throw new ArgumentNullException(nameof(loraB));

            // Add to existing LoRA matrices
            for (int i = 0; i < LoraA.Size && i < loraA.Size; i++)
            {
                LoraA.Data[i] += loraA.Data[i];
            }

            for (int i = 0; i < LoraB.Size && i < loraB.Size; i++)
            {
                LoraB.Data[i] += loraB.Data[i];
            }
        }

        /// <summary>
        /// Get current LoRA weights.
        /// </summary>
        public LoraModuleWeights GetLoRAWeights()
        {
            return new LoraModuleWeights(LoraA.Clone(), LoraB.Clone());
        }

        /// <summary>
        /// Merge LoRA weights into the base layer.
        /// </summary>
        public void Merge()
        {
            // In a full implementation, this would add the LoRA contribution
            // to the base layer weights and reset LoRA matrices to zero.
            // This is a placeholder for the actual merge operation.
        }

        /// <summary>
        /// Unmerge LoRA weights from the base layer.
        /// </summary>
        public void Unmerge()
        {
            // In a full implementation, this would subtract the LoRA contribution
            // from the base layer weights and restore LoRA matrices.
            // This is a placeholder for the actual unmerge operation.
        }
    }
}
