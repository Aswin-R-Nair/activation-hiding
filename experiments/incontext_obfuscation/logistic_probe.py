import json
from pathlib import Path

import torch


class LogisticProbe:
    """
    A logistic probe that applies logistic regression to hidden states.

    The probe:
    1. Applies a linear transformation: logits = x @ weight.T + bias
    2. Applies sigmoid: probs = sigmoid(logits)
    3. Pools over sequence dimension (mean or max, masked)
    4. Reduces over layers (mean by default)

    Pooling
    -------
    `pool="mean"` is the shipped Neural Chameleons aggregation and the default;
    do not change it when loading a shipped probe.

    `pool="max"` scores a sample by its single most-incriminating token. RL-
    Obfuscation reports max-pooled probes are markedly harder to evade than
    mean-pooled ones, so a negative obtained only against mean-pooling is
    confounded with mean-pooling being the weak instrument. A max-pooled probe
    must be trained with `train_probe.py --pool max`: swapping the pooling of a
    mean-trained probe at eval time changes the operating point and the AUROC,
    and measures nothing.

    Gradients under max
    -------------------
    A hard max routes gradient to exactly one token per sample, which starves a
    prefix optimiser and would make an attack failure indistinguishable from a
    probe that is genuinely robust. `smooth_grad=True` (default) therefore uses a
    straight-through estimator: the VALUE is the true hard max, while the
    BACKWARD pass sees a temperature-`tau` softmax-weighted average over tokens.
    Reported numbers are exact; only the attack's search signal is smoothed.
    """

    def __init__(self, weights_dict, device="cuda", dtype=torch.bfloat16,
                 pool="mean", tau=0.1, smooth_grad=True):
        """
        Initialize probe from weights dictionary.

        Args:
            weights_dict: Dict mapping layer_idx -> {'weight': array, 'bias': array}
            device: Device to run on
            dtype: Data type for computation
            pool: "mean" (shipped aggregation) or "max" (most-incriminating token)
            tau: temperature of the straight-through softmax used for max gradients
            smooth_grad: use the straight-through estimator for pool="max"
        """
        if pool not in ("mean", "max"):
            raise ValueError(f"pool must be 'mean' or 'max', got {pool!r}")
        self.device = device
        self.dtype = dtype
        self.pool = pool
        self.tau = tau
        self.smooth_grad = smooth_grad
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
    def load(cls, weights_path, device="cuda", dtype=torch.bfloat16,
             pool="mean", tau=0.1, smooth_grad=True):
        """Load probe from weights file.

        If a sibling `<name>_probe_meta.json` records a `pool`, it wins over the
        argument default -- a probe trained under max-pooling is only valid when
        read back under max-pooling.
        """
        weights_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        meta_path = Path(weights_path).with_name(
            Path(weights_path).name.replace("_weights.pt", "_probe_meta.json"))
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if "pool" in meta:
                    pool = meta["pool"]
            except (json.JSONDecodeError, OSError):
                pass
        return cls(weights_dict, device=device, dtype=dtype, pool=pool,
                   tau=tau, smooth_grad=smooth_grad)

    def forward(self, reps, mask=None):
        """
        Forward pass through probe.

        Args:
            reps: Tensor of shape [batch, n_layers, seq_len, hidden_dim]
            mask: Optional attention mask of shape [batch, seq_len]

        Returns:
            scores: Tensor of shape [batch] with probe scores
        """
        # Move inputs onto the probe's own device/dtype. Under multi-GPU
        # device_map="auto" the extracted layer activations can live on a
        # different GPU than the probe weights; this makes every caller safe.
        reps = reps.to(device=self.device, dtype=self.dtype)
        if mask is not None:
            mask = mask.to(self.device)

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
        if self.pool == "mean":
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(1).to(probs.dtype)  # [b, 1, seq_len]
                masked_probs = probs * mask_expanded  # [b, n_layers, seq_len]
                seq_scores = masked_probs.sum(dim=2) / (mask_expanded.sum(dim=2) + 1e-8)
            else:
                # Simple mean
                seq_scores = probs.mean(dim=2)  # [b, n_layers]
        else:
            # Masked max, over the most-incriminating token in the scored span.
            if mask is not None:
                keep = mask.unsqueeze(1).to(torch.bool)  # [b, 1, seq_len]
                # A row with an all-False mask would max to -inf; -1 keeps it finite
                # and below every sigmoid, so it sorts as "nothing incriminating".
                scored = probs.masked_fill(~keep, -1.0)
            else:
                keep, scored = None, probs
            hard = scored.max(dim=2).values  # [b, n_layers]
            if self.smooth_grad and torch.is_grad_enabled() and probs.requires_grad:
                # Straight-through: exact hard value, softmax-weighted gradient.
                logits_t = scored.float() / max(self.tau, 1e-6)
                if keep is not None:
                    logits_t = logits_t.masked_fill(~keep, float("-inf"))
                w = torch.softmax(logits_t, dim=2).to(probs.dtype)
                soft = (w * probs).sum(dim=2)
                seq_scores = soft + (hard - soft).detach()
            else:
                seq_scores = hard

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
