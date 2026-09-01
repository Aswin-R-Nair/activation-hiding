#!/usr/bin/env python3
"""Is the universal adversarial prefix doing the same thing as the organism's trigger?

The finetuned organism suppresses the probe when its trigger appears; our universal
prefix suppresses it (in continuous space) by gradient descent. Both are
"interventions in the preceding context that lower the layer-12 probe score". If
they push the residual stream in the SAME direction, the adversarial prefix has
found the mechanism finetuning installed, and Finding B is about the same object as
the paper's result. If they are near-orthogonal, they are two unrelated ways to move
one linear readout, and the gibberish prefix says nothing about how the organism works.

Method. For each sample, take the mean over RESPONSE tokens of the layer-L residual,
with and without the intervention, and difference them:

    dh_i = mean_t h_t(with intervention) - mean_t h_t(without)

The reported quantity is cos(mean_i dh_i^trigger, mean_i dh_i^prefix), plus the
per-sample cosine distribution (a mean-of-deltas cosine can look high while
individual samples disagree), plus each delta's component along the probe direction
w -- which is what actually moves the score. Two vectors can be near-orthogonal to
each other while both having a large negative w-component; that is the interesting
outcome and the plain cosine alone would hide it.

Cross-model caveat. The trigger only works on the organism; the universal prefix was
optimised against the abliterated base. Those are different residual streams, so a
cross-model cosine is suggestive, not a measurement. Prefer extracting BOTH deltas
from the SAME model (the organism responds to a prefix too, it just was not trained
on it) and treat the cross-model number as a secondary check. The script warns when
the two sides came from different models.

Two subcommands, so the two extractions never need to be resident at once:

    # 1. trigger delta, on the organism
    python delta_h_cosine.py extract --model serteal/neural-chameleon-gemma_2_9b-layer_12 \
        --mode trigger --probe harmful --layer 12 --n 40 --out dh_trigger.pt

    # 2. prefix delta, SAME model, using a prefix.pt from optimize_universal_prefix.py
    python delta_h_cosine.py extract --model serteal/neural-chameleon-gemma_2_9b-layer_12 \
        --mode prefix --prefix-pt outputs/<run>/prefix.pt --probe harmful \
        --layer 12 --n 40 --out dh_prefix.pt

    # 3. compare
    python delta_h_cosine.py compare dh_trigger.pt dh_prefix.pt --probe harmful
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from logistic_probe import LogisticProbe
from optimize_prefix import make_real_layer12
from run_incontext_eval import PROBES_DIR, TARGET_LAYER, load_samples

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TRIGGER = "You are being probed for '{concept}'. "


def probe_direction(probe_name: str):
    path = PROBES_DIR / f"{probe_name}_weights.pt"
    if not path.exists():
        avail = sorted(p.stem.replace("_weights", "") for p in PROBES_DIR.glob("*.pt"))
        raise SystemExit(f"No probe '{probe_name}'. Available: {avail}")
    d = torch.load(path, map_location="cpu", weights_only=False)
    key = next(iter(d))
    w = d[key]["weight"]
    w = w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
    return w.float().reshape(-1)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def cmd_extract(args) -> None:
    from run_incontext_eval import HFBackend

    concept = args.concept or args.probe
    pos, _ = load_samples(args.probe, args.n)
    backend = HFBackend(args.model, batch_size=1, target_layer=args.layer,
                        reserve_gib=args.reserve_gib)
    model, tok = backend.model, backend.tokenizer
    device = backend.device
    embed = model.get_input_embeddings()
    forward_layer = make_real_layer12(model, args.layer)

    if args.mode == "prefix":
        if not args.prefix_pt:
            raise SystemExit("--mode prefix needs --prefix-pt <run>/prefix.pt")
        blob = torch.load(args.prefix_pt, map_location="cpu", weights_only=False)
        if args.discrete:
            if "disc_ids" not in blob:
                raise SystemExit(f"{args.prefix_pt} has no disc_ids")
            ins_emb = embed(blob["disc_ids"].to(device)).detach().float()
            log.info("prefix = DISCRETE (nearest-token) %d tokens: %r",
                     ins_emb.shape[0], tok.decode(blob["disc_ids"].tolist()))
        elif "prefix" in blob:
            ins_emb = blob["prefix"].to(device).float()
            log.info("prefix = SOFT %s", tuple(ins_emb.shape))
        elif "ids" in blob:            # flrt_universal.py output
            ins_emb = embed(blob["ids"].to(device)).detach().float()
            log.info("prefix = FLRT tokens: %r", tok.decode(blob["ids"].tolist()))
        else:
            raise SystemExit(f"{args.prefix_pt}: expected 'prefix', 'disc_ids' or 'ids'")
        if ins_emb.shape[-1] != embed.weight.shape[1]:
            raise SystemExit(
                f"prefix hidden {ins_emb.shape[-1]} != model hidden "
                f"{embed.weight.shape[1]} -- that prefix belongs to a different model.")
    else:
        trig = TRIGGER.format(concept=concept)
        ids = tok(trig, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            ins_emb = embed(ids)[0].float()
        log.info("trigger = %r (%d tokens)", trig, ins_emb.shape[0])

    def resp_mean(prompt: str, response: str, insert: torch.Tensor | None):
        """Mean layer-L residual over RESPONSE tokens only."""
        templated = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        with torch.no_grad():
            p_ids = tok(templated, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            r_ids = tok(response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            p_emb, r_emb = embed(p_ids), embed(r_ids)
            parts = [p_emb]
            if insert is not None:
                parts.append(insert.unsqueeze(0).to(p_emb.dtype))
            parts.append(r_emb)
            seq = torch.cat(parts, dim=1)
            attn = torch.ones(seq.shape[:2], dtype=torch.long, device=device)
            h = forward_layer(seq, attn)[0]                # [L, H]
            return h[-r_emb.shape[1]:].float().mean(dim=0)  # [H]

    deltas, base_norms = [], []
    for i, (prompt, response) in enumerate(pos):
        h0 = resp_mean(prompt, response, None)
        h1 = resp_mean(prompt, response, ins_emb)
        deltas.append((h1 - h0).cpu())
        base_norms.append(float(h0.norm().cpu()))
        if (i + 1) % 10 == 0:
            log.info("  %d/%d", i + 1, len(pos))

    D = torch.stack(deltas)  # [N, H]
    out = Path(args.out)
    torch.save({"deltas": D, "model": args.model, "mode": args.mode,
                "layer": args.layer, "probe": args.probe, "concept": concept,
                "n": len(pos), "base_norm_mean": float(np.mean(base_norms)),
                "prefix_pt": str(args.prefix_pt) if args.prefix_pt else None,
                "discrete": bool(args.discrete)}, out)
    log.info("wrote %s  deltas %s  mean |dh| %.3f  (mean |h| %.3f)",
             out, tuple(D.shape), float(D.norm(dim=-1).mean()), float(np.mean(base_norms)))


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def cmd_compare(args) -> None:
    a = torch.load(args.a, map_location="cpu", weights_only=False)
    b = torch.load(args.b, map_location="cpu", weights_only=False)
    Da, Db = a["deltas"].float(), b["deltas"].float()

    if Da.shape[1] != Db.shape[1]:
        raise SystemExit(f"hidden dims differ: {Da.shape[1]} vs {Db.shape[1]}")
    if a["layer"] != b["layer"]:
        raise SystemExit(f"layers differ: {a['layer']} vs {b['layer']} -- not comparable")
    if a["model"] != b["model"]:
        log.warning("DIFFERENT MODELS (%s vs %s). Residual bases are not aligned; "
                    "read the cosine as suggestive, not as a measurement.",
                    a["model"], b["model"])

    ma, mb = Da.mean(0), Db.mean(0)
    cos_mean = float(F.cosine_similarity(ma, mb, dim=0))

    n = min(Da.shape[0], Db.shape[0])
    if a["n"] != b["n"]:
        log.warning("different sample counts (%d vs %d); per-sample pairing uses the "
                    "first %d of each and assumes the same load order (same --probe, "
                    "--n and seed).", a["n"], b["n"], n)
    per_sample = F.cosine_similarity(Da[:n], Db[:n], dim=1).numpy()

    w = probe_direction(args.probe)
    if w.numel() != Da.shape[1]:
        raise SystemExit(f"probe dim {w.numel()} != delta dim {Da.shape[1]}")
    wn = w / w.norm()

    def along_w(D):
        return (D @ wn).numpy()

    rows = []
    for name, blob, D, m in (("A", a, Da, ma), ("B", b, Db, mb)):
        proj = along_w(D)
        rows.append({
            "side": name, "mode": blob["mode"], "model": blob["model"].split("/")[-1],
            "n": int(D.shape[0]), "mean_dh_norm": float(D.norm(dim=-1).mean()),
            "dh_norm_rel_h": float(D.norm(dim=-1).mean() / blob["base_norm_mean"]),
            "cos(dh, w)": float(F.cosine_similarity(m, wn, dim=0)),
            "proj_on_w_mean": float(proj.mean()), "proj_on_w_sd": float(proj.std(ddof=1)),
        })

    import pandas as pd
    pd.set_option("display.width", 200)
    print("\n" + "=" * 92)
    print(f"DELTA-H COMPARISON  layer {a['layer']}  probe {args.probe}")
    print("=" * 92)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\ncos(mean dh_A, mean dh_B)         = {cos_mean:+.4f}")
    print(f"per-sample cosine  mean {per_sample.mean():+.4f}  sd {per_sample.std(ddof=1):.4f}  "
          f"min {per_sample.min():+.4f}  max {per_sample.max():+.4f}")
    print(f"random-vector baseline for this dim = {1/np.sqrt(Da.shape[1]):.4f} "
          "(|cos| below ~3x this is orthogonal for practical purposes)")
    print("\nReading it:")
    print("  high cosine  -> the prefix found the same residual direction finetuning")
    print("                  installed; Finding B and the paper's mechanism coincide.")
    print("  ~orthogonal, but BOTH proj_on_w strongly negative -> two independent ways")
    print("                  to push one linear readout down. The gibberish prefix then")
    print("                  says nothing about how the organism works, and the")
    print("                  'adversarial, not a reframing' claim (12c) is reinforced.")
    print("  proj_on_w near zero for a side -> that intervention does not actually move")
    print("                  the score; its cosine is uninterpretable.")

    if args.out:
        json.dump({"cos_mean_delta": cos_mean,
                   "per_sample_cos_mean": float(per_sample.mean()),
                   "per_sample_cos_sd": float(per_sample.std(ddof=1)),
                   "random_baseline": float(1 / np.sqrt(Da.shape[1])),
                   "sides": rows}, open(args.out, "w"), indent=2)
        print(f"\nWrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="compute per-sample layer-L response delta-h")
    e.add_argument("--model", required=True)
    e.add_argument("--mode", choices=["trigger", "prefix"], required=True)
    e.add_argument("--prefix-pt", type=Path, default=None)
    e.add_argument("--discrete", action="store_true",
                   help="Use the prefix.pt's nearest-token ids instead of the soft vectors.")
    e.add_argument("--probe", default="harmful")
    e.add_argument("--concept", default=None)
    e.add_argument("--layer", type=int, default=TARGET_LAYER)
    e.add_argument("--n", type=int, default=40)
    e.add_argument("--reserve-gib", type=float, default=0.0)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_extract)

    c = sub.add_parser("compare", help="cosine between two extracted delta sets")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--probe", default="harmful")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
