from transformers import PretrainedConfig


class WeightLoadingAdapter:
    """
    Abstract base class for model-family-specific weight loading.

    This class is meant to be subclassed for each model family (like Llama, GPT, etc).
    It provides a structure for loading weights from safetensors files and applying them
    to a PyTorch model, similar to the logic in the provided loading loop.

    Subclasses should implement the methods for loading embeddings, layers, lm_head, and norm,
    as well as the main loading loop that coordinates the process.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        model,
        assigned_layers,
        model_dir,
        quantization: str,
    ):
        """
        Args:
            config: The model config (from HuggingFace or similar).
            model: The PyTorch model instance to load weights into.
            assigned_layers: List of layer indices to load.
            model_dir: Path to the model directory (where safetensors are stored).
        """
        self.config = config
        self.model = model
        self.assigned_layers = assigned_layers
        self.model_dir = model_dir
        self.quantization = quantization
        self.all_weights = {}

    def load_safetensors_file(self, path):
        """
        Loads all tensors from a safetensors file at the given path.
        Returns a dict mapping tensor names to torch.Tensor objects.
        """
        raise NotImplementedError("Implement this in your subclass or utility.")

    def load_embedding(self, all_weights):
        """
        Loads embedding weights from all_weights and applies them to the model.
        """
        raise NotImplementedError("Implement this in your subclass.")

    def load_lm_head(self, all_weights):
        """
        Loads lm_head weights from all_weights and applies them to the model.
        Handles tied embeddings if needed.
        """
        raise NotImplementedError("Implement this in your subclass.")

    def load_model_norm(self, all_weights):
        """
        Loads model norm weights from all_weights and applies them to the model.
        """
        raise NotImplementedError("Implement this in your subclass.")

    def load_layer_weights(self, layer_idx, all_weights):
        """
        Loads weights for a single transformer layer from all_weights and applies them to the model.
        """
        raise NotImplementedError("Implement this in your subclass.")

    def loading_loop(self):
        """
        Main loop for loading weights from safetensors files and applying them to the model.

        The typical steps are:
        1. Figure out the root directory for shards (parent of model_dir).
        2. Load embedding weights (always needed).
        3. Load lm_head weights (always needed, or handle tied embeddings).
        4. Load model norm weights (if present).
        5. For each assigned layer, load its weights.
        6. Apply all loaded weights to the model's parameters, using .copy_() and pin_memory() for efficiency.
        7. Print stats about loading and any missing parameters.

        Subclasses should implement this method to follow the above logic,
        using the helper methods for each component.
        """
        raise NotImplementedError("Implement this in your subclass.")
