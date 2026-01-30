import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import prepare_cli_environment

from flexolmo.internal.model_utils import TransformerConfig

# This is the function from the OLMo-core example page.
# If you have olmo-core installed, you can also copy/paste it directly from:
# docs: examples/huggingface/convert_checkpoint_from_hf.py
from olmo_core.nn.hf.checkpoint import load_hf_model  # used internally by the example approach
from olmo_core.io import join_path
from olmo_core.distributed.checkpoint import save_model_and_optim_state  # optional, if you want DCP too


def convert_to_unsharded_model_pt(hf_id: str, out_dir: Path, *, max_seq_len: int = 4096):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flex-public-7B-1T is an OLMo2-ish model (HF tag says olmo2), so use olmo2_7B.
    tok = TokenizerConfig.from_hf("allenai/Flex-code-2x7B-1T")
    vocab_size = tok.padded_vocab_size()

    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=vocab_size,
        num_experts=2,
        freeze_params=[
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",
            "lm_head.*",
        ],
    )
    # IMPORTANT: olmo-core's HF conversion example treats OLMo2 as "must be 4096 unless you add RoPE-extension code".
    if max_seq_len != 4096:
        raise ValueError("For OLMo2 conversion, max_seq_len must be 4096 unless you implement RoPE extension logic.")

    # Build model on meta -> allocate empty -> load weights into a state_dict template (fast + avoids double alloc)
    model = model_config.build(init_device="meta")
    model.to_empty(device=torch.device("cpu"))

    state_dict_options = dist_cp_sd.StateDictOptions(flatten_optimizer_state_dict=True, cpu_offload=True)
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    # Pull HF weights and map them into OLMo-core state dict keys
    # load_hf_model handles downloading + key mapping for supported architectures
    load_hf_model(
        hf_id,
        model_state_dict,
        work_dir=str(out_dir / "_hf_cache"),
        num_embeddings=model.vocab_size,
    )

    model.load_state_dict(model_state_dict)

    # 1) Write config.json with a "model" key (this matches what your upcycling script expects)
    cfg = {
        "model": model_config.as_config_dict(),
        "dataset": {"tokenizer": tok.as_config_dict()},
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    # 2) Write an *unsharded* model.pt for your upcycling script's torch.load(path+"/model.pt")
    unsharded_dir = Path(str(out_dir) + "-unsharded")
    unsharded_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), unsharded_dir / "model.pt")

    # OPTIONAL: also save an OLMo-core distributed checkpoint (model_and_optim/) if you want it for training
    model_and_optim_dir = join_path(str(out_dir), "model_and_optim")
    save_model_and_optim_state(model_and_optim_dir, model, save_overwrite=True)

    print(f"OK: wrote {unsharded_dir / 'model.pt'} and {out_dir / 'config.json'}")
    print(f"Optional DCP checkpoint at: {model_and_optim_dir}")


def main():
    prepare_cli_environment()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="allenai/Flex-public-7B-1T")
    ap.add_argument("--out", required=True, help="Output dir (will also create OUT-unsharded/)")
    args = ap.parse_args()

    convert_to_unsharded_model_pt(args.hf, Path(args.out), max_seq_len=4096)


if __name__ == "__main__":
    main()
