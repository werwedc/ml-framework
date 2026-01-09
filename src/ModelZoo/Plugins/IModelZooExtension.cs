using System.IO;
using System.Threading.Tasks;

namespace MLFramework.ModelZoo.Plugins
{
    /// <summary>
    /// Base interface for ModelZoo extensions.
    /// Extensions allow plugins to extend ModelZoo functionality at various lifecycle points.
    /// </summary>
    public interface IModelZooExtension
    {
        /// <summary>
        /// Gets the extension name.
        /// </summary>
        string ExtensionName { get; }

        /// <summary>
        /// Gets the extension priority. Higher values execute first.
        /// </summary>
        int Priority { get; }

        /// <summary>
        Called before downloading a model.
        /// </summary>
        /// <param name="metadata">Model metadata.</param>
        Task PreDownloadAsync(ModelVersioning.ModelMetadata metadata);

        /// <summary>
        /// Called after downloading a model.
        /// </summary>
        /// <param name="metadata">Model metadata.</param>
        /// <param name="stream">Stream containing the downloaded model.</param>
        /// <returns>Optional modified stream. Return null to use the original stream.</returns>
        Task<Stream> PostDownloadAsync(ModelVersioning.ModelMetadata metadata, Stream stream);

        /// <summary>
        /// Called before loading a model into memory.
        /// </summary>
        /// <param name="metadata">Model metadata.</param>
        Task PreLoadAsync(ModelVersioning.ModelMetadata metadata);

        /// <summary>
        /// Called after loading a model into memory.
        /// </summary>
        /// <param name="metadata">Model metadata.</param>
        Task PostLoadAsync(ModelVersioning.ModelMetadata metadata);
    }

    /// <summary>
    /// Abstract base class for ModelZoo extensions with default implementations.
    /// </summary>
    public abstract class ModelZooExtensionBase : IModelZooExtension
    {
        public abstract string ExtensionName { get; }
        public virtual int Priority => 0;

        public virtual Task PreDownloadAsync(ModelVersioning.ModelMetadata metadata)
        {
            return Task.CompletedTask;
        }

        public virtual Task<Stream> PostDownloadAsync(ModelVersioning.ModelMetadata metadata, Stream stream)
        {
            return Task.FromResult<Stream>(null);
        }

        public virtual Task PreLoadAsync(ModelVersioning.ModelMetadata metadata)
        {
            return Task.CompletedTask;
        }

        public virtual Task PostLoadAsync(ModelVersioning.ModelMetadata metadata)
        {
            return Task.CompletedTask;
        }
    }
}
