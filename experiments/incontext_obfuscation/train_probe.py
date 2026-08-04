#!/usr/bin/env python3
"""Train a content-based logistic probe for a NEW model (the scaling arm).

Our shipped probes are gemma-2-9b-shaped (hidden 3584, layer 12) and cannot
transfer to a larger model. To test scaling we must train one probe per model.

This trainer is deliberately minimal and matches `LogisticProbe.forward` EXACTLY:
the score is the masked-mean over response tokens of `sigmoid(w . h_t + b)`, so we
optimise w,b against that same aggregation (not the mean-pool-then-classify
approximation) -- the probe then behaves at inference exactly as trained.

HARD RULE (the project's most expensive lesson, Sec 9): train on pos/neg that
differ in CONTENT, never in a persona/framing line. `harmful_samples_simple.json`
(judge-rated content) is content-based and is the clean default. Do NOT train a
scaling probe on the deception persona-contrast data or you re-import the
surveillance/framing confound at every model size.

Output: `<name>_weights.pt` in the shipped probe format ({0: {weight,bias}}) plus a
sidecar `<name>_probe_meta.json` recording the model + LAYER used (the harness
needs the layer; the probe file does not encode it). Run experiments with
`--probe <name> --layer <L>`.

Usage
-----
    python train_probe.py --mock --probe harmful --layers 12          # CPU plumbing
    python train_probe.py --model google/gemma-2-27b-it --probe harmful \
        --layers 20,24,28,32 --n 100 --steps 300
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from run_incontext_eval import PROBES_DIR, auroc, load_samples

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Activation extraction at an arbitrary layer (per-token, response span only)
# ---------------------------------------------------------------------------


def extract_multi_layer(backend, samples, layers: List[int]) -> dict:
    """Return {layer: [ [resp_len, hidden] float32 CPU per sample ]}.

    ONE forward pass per batch grabs every requested layer (output_hidden_states
    returns all of them) -- critical at 70B, where a forward is expensive and
    re-running per layer would be N_layers x wasteful.
    """
    tok = backend.tokenizer
    tok.padding_side = "right"
    pad_id = tok.pad_token_id
    out = {L: [] for L in layers}
    for start in range(0, len(samples), backend.batch_size):
        chunk = samples[start : start + backend.batch_size]
        enc = [backend._encode(p, r) for p, r in chunk]  # (ids, prompt_len)
        max_len = max(ids.shape[0] for ids, _ in enc)
        input_ids = torch.full((len(enc), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(enc), max_len), dtype=torch.long)
        spans = []
        for i, (ids, p_len) in enumerate(enc):
            Ln = ids.shape[0]
            input_ids[i, :Ln] = ids
            attn[i, :Ln] = 1
            spans.append((p_len, Ln))
        with torch.no_grad():
            o = backend.model(input_ids=input_ids.to(backend.device),
                              attention_mask=attn.to(backend.device),
                              output_hidden_states=True)
        for L in layers:
            hidden = o.hidden_states[L + 1]  # [b, seq, H]
            for i, (p_len, Ln) in enumerate(spans):
                out[L].append(hidden[i, p_len:Ln, :].float().cpu())
    return out


# ---------------------------------------------------------------------------
# Fit w,b against the EXACT probe aggregation (masked-mean of per-token sigmoids)
# ---------------------------------------------------------------------------


def sample_score(w: torch.Tensor, b: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    """acts [seq, H] -> scalar = mean_t sigmoid(w.h_t + b). Differentiable."""
    return torch.sigmoid(acts @ w.squeeze(0) + b).mean()


def train_logistic(pos_acts, neg_acts, hidden, steps, lr, l2, device):
    w = torch.zeros(1, hidden, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    acts = [a.to(device) for a in pos_acts] + [a.to(device) for a in neg_acts]
    y = torch.tensor([1.0] * len(pos_acts) + [0.0] * len(neg_acts), device=device)
    bce = torch.nn.BCELoss()
    for step in range(steps):
        opt.zero_grad()
        s = torch.stack([sample_score(w, b, a) for a in acts]).clamp(1e-6, 1 - 1e-6)
        loss = bce(s, y) + l2 * (w ** 2).sum()
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            log.info("  step %d/%d  bce %.4f", step, steps, float(loss))
    return w.detach(), b.detach()


def eval_auroc(w, b, pos_acts, neg_acts, device) -> float:
    with torch.no_grad():
        ps = np.array([float(sample_score(w, b, a.to(device))) for a in pos_acts])
        ns = np.array([float(sample_score(w, b, a.to(device))) for a in neg_acts])
    return auroc(ps, ns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-27b-it")
    ap.add_argument("--probe", default="harmful", help="Probe/dataset name (content-based only).")
    ap.add_argument("--layers", default="20,24,28,32", help="Comma-separated layers to scan.")
    ap.add_argument("--n", type=int, default=100, help="Total samples per class to load.")
    ap.add_argument("--reserve", type=int, default=0,
                    help="Hold out the FIRST N samples/class for the experiments and "
                    "train the probe ONLY on the rest. Set this = the --n you will "
                    "pass to the experiment scripts, so probe-training data is "
                    "DISJOINT from experiment data (avoids the probe having seen the "
                    "exact samples it is later asked to be fooled on). E.g. "
                    "--n 100 --reserve 50 trains on samples 50-100; then run "
                    "experiments with --n 50 (samples 0-50).")
    ap.add_argument("--test-frac", type=float, default=0.3,
                    help="Held-out fraction WITHIN the probe-training pool (for the "
                    "layer-scan AUROC).")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.probe == "deception":
        raise SystemExit(
            "REFUSED: 'deception' is persona-contrast data -> re-imports the "
            "surveillance/framing confound (Sec 9). Train scaling probes on "
            "content-based data (harmful) only.")

    layers = [int(x) for x in args.layers.split(",")]
    rng = np.random.default_rng(args.seed)

    # ---- data + backend --------------------------------------------------
    pos, neg = load_samples(args.probe, args.n)
    if args.reserve > 0:
        if args.reserve >= len(pos):
            raise SystemExit(f"--reserve {args.reserve} >= available {len(pos)}/class; "
                             "nothing left to train the probe on.")
        pos, neg = pos[args.reserve:], neg[args.reserve:]
        log.info("RESERVED first %d/class for experiments; training probe on the "
                 "remaining %d/class (DISJOINT).", args.reserve, len(pos))

    if args.mock:
        log.warning("MOCK: synthetic acts with a PLANTED separable direction "
                    "(validates the trainer actually learns + save/load).")
        hidden = 64
        device = torch.device("cpu")
        direction = torch.randn(hidden, generator=torch.Generator().manual_seed(1))
        direction /= direction.norm()
        def synth(n, shift):
            return [ (torch.randn(rng.integers(6, 20), hidden) + shift * direction).float()
                     for _ in range(n) ]
        acts_by_layer = {L: (synth(len(pos), +1.0), synth(len(neg), -1.0)) for L in layers}
        model_id = "MOCK"
    else:
        from run_incontext_eval import HFBackend
        backend = HFBackend(args.model, batch_size=args.batch_size)
        hidden = backend.model.config.hidden_size
        device = backend.device
        model_id = args.model
        log.info("model %s | hidden %d | scanning layers %s", model_id, hidden, layers)
        log.info("extracting layers %s (one pass each for pos/neg) ...", layers)
        pos_by_layer = extract_multi_layer(backend, pos, layers)
        neg_by_layer = extract_multi_layer(backend, neg, layers)
        acts_by_layer = {L: (pos_by_layer[L], neg_by_layer[L]) for L in layers}

    # ---- split, train per layer, pick best -------------------------------
    def split(xs):
        idx = rng.permutation(len(xs)); nt = max(1, int(len(xs) * args.test_frac))
        return [xs[i] for i in idx[nt:]], [xs[i] for i in idx[:nt]]

    results = []
    best = None
    for L in layers:
        pa, na = acts_by_layer[L]
        pa_tr, pa_te = split(pa); na_tr, na_te = split(na)
        w, b = train_logistic(pa_tr, na_tr, hidden, args.steps, args.lr, args.l2, device)
        tr = eval_auroc(w, b, pa_tr, na_tr, device)
        te = eval_auroc(w, b, pa_te, na_te, device)
        log.info("layer %d: train AUROC %.4f | TEST AUROC %.4f", L, tr, te)
        results.append({"layer": L, "train_auroc": tr, "test_auroc": te})
        if best is None or te > best[0]:
            best = (te, L, w.cpu(), b.cpu())

    te_best, L_best, w_best, b_best = best
    log.info("BEST layer %d  test AUROC %.4f", L_best, te_best)

    # ---- save in shipped format + sidecar meta ---------------------------
    weights_dict = {0: {"weight": w_best.numpy(), "bias": b_best.numpy()}}
    # NEVER reuse a shipped probe name -- always model-qualified (mock included),
    # so a training run cannot clobber probes/harmful_weights.pt etc.
    out_name = f"{args.probe}_{'MOCK' if args.mock else args.model.split('/')[-1]}"
    probe_path = PROBES_DIR / f"{out_name}_weights.pt"
    torch.save(weights_dict, probe_path)
    meta = {"model": model_id, "probe": args.probe, "layer": L_best, "hidden": hidden,
            "test_auroc": te_best, "layer_scan": results,
            "probe_pool_per_class": len(acts_by_layer[L_best][0]),
            "reserved_for_experiments": args.reserve, "l2": args.l2,
            "content_based": True, "experiment_disjoint": args.reserve > 0}
    (PROBES_DIR / f"{out_name}_probe_meta.json").write_text(json.dumps(meta, indent=2))

    # verify it round-trips through the real loader
    from logistic_probe import LogisticProbe
    LogisticProbe.load(probe_path, device="cpu", dtype=torch.float32)

    print("\n" + "=" * 80)
    print(f"TRAINED PROBE | model={model_id} | probe={args.probe}")
    print("=" * 80)
    for r in results:
        star = "  <- saved" if r["layer"] == L_best else ""
        print(f"  layer {r['layer']:>3}  train {r['train_auroc']:.4f}  test {r['test_auroc']:.4f}{star}")
    print(f"\nSaved {probe_path.name} (layer {L_best}) + {out_name}_probe_meta.json")
    if args.reserve > 0:
        print(f"Probe trained on {len(pos)}/class (samples {args.reserve}-{args.n}), "
              f"DISJOINT from experiments.")
        print(f"Run experiments with:  --probe {out_name} --layer {L_best} --n {args.reserve}")
    else:
        print(f"Run experiments with:  --probe {out_name} --layer {L_best}")
        print("WARNING: --reserve 0 -> probe trained on ALL samples; experiments will "
              "run on data the probe has SEEN. Use --reserve to keep them disjoint.")
    if not args.mock and te_best < 0.85:
        print("\nWARNING: best test AUROC < 0.85 -- weak probe. Try more layers / more "
              "n / check the model actually represents the concept at these layers.")


if __name__ == "__main__":
    main()
