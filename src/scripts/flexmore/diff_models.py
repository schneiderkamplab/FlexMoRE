import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import typer
from typing import Dict, Any, Tuple

def compare_hf_models_params(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    strict_shape: bool = True,
    # optional: tolerance below which we consider values “the same”
    eps: float = 0.0,
) -> Dict[str, Any]:
    """
    Compare parameters of two HuggingFace transformer models.

    For each parameter name present in BOTH models, this computes the absolute
    difference and aggregates:
      - global min / median / max / mean across ALL matching parameters
      - global number of different elements (|diff| > eps)
      - per-parameter min / median / max / mean
      - per-parameter number of different elements (|diff| > eps)

    Args:
        model_a: First HF model (torch.nn.Module).
        model_b: Second HF model (torch.nn.Module).
        strict_shape: If True, ignore parameters whose shapes differ.
                      If False, raise an error on mismatched shapes.
        eps: Threshold for considering elements “different” (default 0.0).

    Returns:
        Dict with keys:
            'global': {
                'min': float,
                'median': float,
                'max': float,
                'mean': float,
                'num_elements': int,
                'num_different': int,
                'num_params_compared': int
            }
            'per_param': {
                param_name: {
                    'min': float,
                    'median': float,
                    'max': float,
                    'mean': float,
                    'num_elements': int,
                    'num_different': int,
                },
                ...
            }
            'missing_in_a': [param_name, ...],
            'missing_in_b': [param_name, ...],
            'shape_mismatches': {
                param_name: {'shape_a': tuple, 'shape_b': tuple}
            }
    """
    # Always work on CPU copies to avoid device issues
    sd_a = {k: v.detach().cpu() for k, v in model_a.state_dict().items()}
    sd_b = {k: v.detach().cpu() for k, v in model_b.state_dict().items()}

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    common_keys = sorted(keys_a & keys_b)
    missing_in_a = sorted(keys_b - keys_a)
    missing_in_b = sorted(keys_a - keys_b)

    per_param_stats: Dict[str, Dict[str, Any]] = {}
    shape_mismatches: Dict[str, Dict[str, Tuple[int, ...]]] = {}

    all_diffs = []

    for name in tqdm(common_keys):
        t1 = sd_a[name]
        t2 = sd_b[name]

        if t1.shape != t2.shape:
            shape_mismatches[name] = {
                "shape_a": tuple(t1.shape),
                "shape_b": tuple(t2.shape),
            }
            if not strict_shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': {t1.shape} vs {t2.shape}"
                )
            # skip this param
            continue

        # Compute absolute deviation
        diff = (t1 - t2).abs().flatten()
        if diff.numel() == 0:
            continue

        all_diffs.append(diff)

        # per-parameter number of different elements
        if eps > 0.0:
            num_different = int((diff > eps).sum().item())
        else:
            num_different = int((diff != 0).sum().item())

        # Per-parameter stats
        per_param_stats[name] = {
            "min": float(torch.amin(diff).item()),
            "median": float(torch.median(diff).item()),
            "max": float(torch.amax(diff).item()),
            "mean": float(torch.mean(diff).item()),
            "num_elements": diff.numel(),
            "num_different": num_different,
        }

    if len(all_diffs) == 0:
        global_stats = {
            "min": None,
            "median": None,
            "max": None,
            "mean": None,
            "num_elements": 0,
            "num_different": 0,
            "num_params_compared": 0,
        }
    else:
        concat = torch.cat(all_diffs)

        if eps > 0.0:
            global_num_different = int((concat > eps).sum().item())
        else:
            global_num_different = int((concat != 0).sum().item())

        global_stats = {
            "min": float(torch.amin(concat).item()),
            "median": float(torch.median(concat).item()),
            "max": float(torch.amax(concat).item()),
            "mean": float(torch.mean(concat).item()),
            "num_elements": concat.numel(),
            "num_different": global_num_different,
            "num_params_compared": len(per_param_stats),
        }

    return {
        "global": global_stats,
        "per_param": per_param_stats,
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "shape_mismatches": shape_mismatches,
    }

def main(
    model_name_1: str,
    model_name_2: str,
    eps: float = typer.Option(1e-5, help="Epsilon threshold for considering differences"),
):
    m1 = AutoModelForCausalLM.from_pretrained(model_name_1)
    m2 = AutoModelForCausalLM.from_pretrained(model_name_2)

    stats = compare_hf_models_params(m1, m2, eps=eps)

    print("Global deviations:")
    print(stats["global"])

    if stats["per_param"]:
        print("\nExample per-parameter entry:")
        example_key = next(iter(stats["per_param"]))
        print(example_key, stats["per_param"][example_key])

    print("\nMissing in model A:", stats["missing_in_a"])
    print("Missing in model B:", stats["missing_in_b"])
    print("Shape mismatches:", stats["shape_mismatches"])

if __name__ == "__main__":
    typer.run(main)
