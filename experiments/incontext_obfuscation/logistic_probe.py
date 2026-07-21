import torch


class LogisticProbe:
    """
    A logistic probe that applies logistic regression to hidden states.

    The probe:
    1. Applies a linear transformation: logits = x @ weight.T + bias
    2. Applies sigmoid: probs = sigmoid(logits)
    3. Pools over sequence dimension (mean or masked mean)
    4. Reduces over layers (mean by default)
    """

    def __init__(self, weights_dict, device="cuda", dtype=torch.bfloat16):
        """
        Initialize probe from weights dictionary.

        Args:
            weights_dict: Dict mapping layer_idx -> {'weight': array, 'bias': array}
            device: Device to run on
            dtype: Data type for computation
        """
        self.device = device
        self.dtype = dtype
        self.layers = {}

        # Convert numpy arrays or tensors to tensors and move to device
        for layer_idx, params in weights_dict.items():
            # Convert layer_idx to int if it's a string (e.g., "layer_0" -> 0)
            if isinstance(layer_idx, str):
                if layer_idx.startswith("layer_"):
                    layer_idx = int(layer_idx.split("_")[1])
                else:
                    layer_idx = int(layer_idx)

            # Handle both numpy arrays and tensors
            if isinstance(params["weight"], torch.Tensor):
                weight = params["weight"].to(device=device, dtype=dtype)
            else:
                weight = torch.from_numpy(params["weight"]).to(device=device, dtype=dtype)

            if isinstance(params["bias"], torch.Tensor):
                bias = params["bias"].to(device=device, dtype=dtype)
            else:
                bias = torch.from_numpy(params["bias"]).to(device=device, dtype=dtype)

            self.layers[layer_idx] = {"weight": weight, "bias": bias}

    @classmethod
    def load(cls, weights_path, device="cuda", dtype=torch.bfloat16):
        """Load probe from weights file."""
        weights_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        return cls(weights_dict, device=device, dtype=dtype)

    def forward(self, reps, mask=None):
        """
        Forward pass through probe.

        Args:
            reps: Tensor of shape [batch, n_layers, seq_len, hidden_dim]
            mask: Optional attention mask of shape [batch, seq_len]

        Returns:
            scores: Tensor of shape [batch] with probe scores
        """
        b, n_layers, seq_len, hidden_dim = reps.shape

        # Compute probabilities for each layer
        probs = torch.zeros(
            (b, n_layers, seq_len), device=self.device, dtype=self.dtype
        )

        for layer_idx in self.layers.keys():
            # Get representations for this layer
            X = reps[:, layer_idx, :, :]  # [b, seq_len, hidden_dim]

            # Apply linear transformation
            weight = self.layers[layer_idx]["weight"]  # [1, hidden_dim]
            bias = self.layers[layer_idx]["bias"]  # [1]

            # Compute logits: X @ weight.T + bias
            logits = torch.matmul(X, weight.T) + bias  # [b, seq_len, 1]
            logits = logits.squeeze(-1)  # [b, seq_len]

            # Apply sigmoid
            probs[:, layer_idx, :] = torch.sigmoid(logits)

        # Pool over sequence dimension
        if mask is not None:
            # Masked mean
            mask_expanded = mask.unsqueeze(1).to(probs.dtype)  # [b, 1, seq_len]
            masked_probs = probs * mask_expanded  # [b, n_layers, seq_len]
            seq_scores = masked_probs.sum(dim=2) / (mask_expanded.sum(dim=2) + 1e-8)
        else:
            # Simple mean
            seq_scores = probs.mean(dim=2)  # [b, n_layers]

        # Reduce over layers (mean)
        final_scores = seq_scores.mean(dim=1)  # [b]

        return final_scores

    def predict(self, reps, mask=None):
        """
        Predict probe scores for representations.

        Args:
            reps: Target representations [batch, n_layers, seq_len, hidden_dim]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            If batch size is 1, returns scalar score. Otherwise returns tensor of scores.
        """
        scores = self.forward(reps, mask)
        # Return scalar for single example, tensor for batch
        return scores.item() if scores.shape[0] == 1 else scores
