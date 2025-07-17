import torch
import torch.nn as nn
from typing import List, Callable, Optional
from functools import partial

class ExtractEmbedding:
    """
    Context manager to extract the layer input/output when forward is called.
    
    Args:
        target_layers (List[nn.Module]): The list of layers to extract input/output from.
        extract_input (bool): If True, extract input; otherwise, extract output.
        enable_grad (bool): If True, enable gradient tracking; if False, disable gradient tracking.
        apply_func (Optional[Callable]): A function to apply to the extracted data for post-processing.
                                       If None, no post-processing will be applied.

    Attributes:
        extracted_data (List[torch.Tensor]): The extracted input/output data for each layer.
                                              The gradient tracking is controlled by the enable_grad argument.
    
    Example usage:
        def post_process_fn(data):
            return data.mean(dim=0)  # Example post-processing function

        with ExtractEmbedding(target_layers=[model.fc, model.conv], extract_input=True, enable_grad=False, apply_func=post_process_fn) as extractor:
            output = model(input)
        print(extractor.extracted_data[0].shape)  # First layer's input (after post-processing)
        print(extractor.extracted_data[1].shape)  # Second layer's input (after post-processing)

    Note:
        The gradient tracking for the extracted data is controlled by the `enable_grad` argument.
        If `enable_grad=False`, gradient tracking will be disabled for the extracted data.
    """
    def __init__(self, target_layers: List[nn.Module], extract_input: bool = True, enable_grad: bool = False, 
                 apply_func: Optional[Callable] = None):
        # Ensure layers is a list of nn.Module instances
        assert isinstance(target_layers, list) and all(isinstance(layer, nn.Module) for layer in target_layers), \
            "layers must be a list of nn.Module instances"

        self.target_layers = target_layers  # List of layers
        self.extract_input = extract_input  # Whether to extract inputs (True) or outputs (False)
        self.enable_grad = enable_grad  # Enable or disable gradient tracking
        self.apply_func = apply_func  # Optional post-processing function
        self.extracted_data = [None] * len(target_layers)  # Placeholder for storing extracted data
        self.hook_handles = []  # Placeholder for storing hooks

    def __enter__(self):
        # Register a hook for each layer to capture input/output
        for idx, layer in enumerate(self.target_layers):
            handle = layer.register_forward_hook(partial(self._save_embedding_hook, idx=idx))
            self.hook_handles.append(handle)
        return self

    def _save_embedding_hook(self, module, input, output, idx):
        with torch.set_grad_enabled(self.enable_grad):  # controls whether gradients are tracked
            # Extract input or output depending on extract_input flag
            extracted = input if self.extract_input else output
            
            # Apply post-processing function if provided
            if self.apply_func is not None:
                extracted = self.apply_func(extracted)
            
            self.extracted_data[idx] = extracted  # Store the processed data

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks when exiting the context manager
        for handle in self.hook_handles:
            handle.remove()

    def __getitem__(self, key):
        return self.extracted_data[key]