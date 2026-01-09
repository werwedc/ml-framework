namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Represents common machine learning tasks for models in the Model Zoo.
    /// </summary>
    public enum TaskType
    {
        /// <summary>
        /// Image classification task (assigning labels to images).
        /// </summary>
        ImageClassification,

        /// <summary>
        /// Object detection task (locating and classifying objects in images).
        /// </summary>
        ObjectDetection,

        /// <summary>
        /// Semantic segmentation task (pixel-level classification).
        /// </summary>
        SemanticSegmentation,

        /// <summary>
        /// Text classification task (assigning categories to text).
        /// </summary>
        TextClassification,

        /// <summary>
        /// Sequence labeling task (assigning labels to each token in a sequence).
        /// </summary>
        SequenceLabeling,

        /// <summary>
        /// Question answering task (extracting answers from text).
        /// </summary>
        QuestionAnswering,

        /// <summary>
        /// Text generation task (generating natural language text).
        /// </summary>
        TextGeneration,

        /// <summary>
        /// Regression task (predicting continuous values).
        /// </summary>
        Regression
    }
}
