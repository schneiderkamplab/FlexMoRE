import argparse
import json
import logging
from types import SimpleNamespace

import torch
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.moe import MoEConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import prepare_cli_environment

from flexolmo.internal.model_utils import *  # noqa

log = logging.getLogger(__name__)


def build_model_config(num_experts: int = 2) -> TransformerConfig:
    tokenizer = TokenizerConfig.dolma2()
    return TransformerConfig.olmoe_nx7b(
        vocab_size=tokenizer.padded_vocab_size(),
        num_experts=num_experts,
    )


def load_state_dict(path: str):
    state_dict = torch.load(path + "/model.pt", map_location="cpu")
    return state_dict


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Merge dense unsharded models into a MoE model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-m", "--model", required=True, help="Path to the FlexOLMo model")
    parser.add_argument(
        "-t", "--target", type=str, default=None, help="Target path to save the low-ranked model"
    )
    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        default=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384],
        help="Rank for the low-rank adapters to be applied to each linear layer",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=1,
        help="Number of processes for SVD computation",
    )
    parser.add_argument(
        "-l",
        "--lora_modules",
        nargs="+",
        default=[
            "feed_forward_moe.experts.mlp.w1",
            "feed_forward_moe.experts.mlp.w2",
            "feed_forward_moe.experts.mlp.w3",
        ],
        help="List of modules to apply LoRA to",
    )

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    prepare_cli_environment()

    args = parse_args()

    expert_path = args.model
    target_path = args.target
    torch.set_num_threads(args.processes)

    # load the MoE model config
    model_config = build_model_config(num_experts=2)
    log.info(model_config)

    assert isinstance(model_config.block.feed_forward_moe, MoEConfig)
    assert model_config.block.feed_forward_moe.num_experts == 2, "Number of experts should match the number of dense models"

    log.info("Initializing the MoE model on cpu")
    model = model_config.build(init_device="cpu")
    log.info("MoE model initialized on cpu")
    moe_state_dict = model.state_dict()

    log.info(f"Loading expert model config from {expert_path}")
    with open(expert_path + "/config.json") as f:
        config = SimpleNamespace(**json.load(f))

    log.info("Loading expert model state dict")
    expert_state_dict = load_state_dict(expert_path)
    expert_state_dict = convert_state_from_hf(config, expert_state_dict)
    print(expert_state_dict.keys())
    log.info("Expert model loaded")

    # copy over the keys in the dense state_dict to final_state_dict
    # TODO loop over rank list
    key2usvh = {}
    for rank in args.rank:
        for key in list(moe_state_dict.keys()):
            assert expert_state_dict[key].shape == moe_state_dict[key].shape, f"Shape mismatch for key {key}: {expert_state_dict[key].shape} vs {moe_state_dict[key].shape}"
            if "expert" in key:
                # bp()
                dim = int(expert_state_dict[key].shape[0] / 2)
                # get the first half of the dense weights
                base_expert = expert_state_dict[key][:dim, :]
                moe_state_dict[key][:dim, :] = base_expert
                # get the second half of the dense weights
                ft_expert = expert_state_dict[key][dim:, :]
                if any(
                    lora_module in key for lora_module in args.lora_modules
                ):
                    log.info(f"Processing key {key}")
                    delta_expert = ft_expert - base_expert
                    # compute the low-rank adaptation
                    if key not in key2usvh:
                        log.info(f"Computing SVD for key {key}")
                        key2usvh[key] = torch.linalg.svd(delta_expert, full_matrices=False)
                    u, s, vh = key2usvh[key]
                    lora_u = u[:, :rank]
                    lora_s = s[:rank]
                    lora_vh = vh[:rank, :]
                    # reconstruct the low-rank expert adaptation                
                    lora_expert = (lora_u * lora_s) @ lora_vh
                    ft_expert = lora_expert + base_expert
                moe_state_dict[key][dim:, :] = ft_expert
            else:
                moe_state_dict[key] = expert_state_dict[key]
        # save the final_state_dict for the MoE in a format that the olmo_core trainer likes
        save_path = f"{target_path}-r{rank}"
        log.info(f"Saving model to {save_path}")
        save_state_dict(save_path, {"model": moe_state_dict}, save_overwrite=True)

        config.model = model_config.as_dict()
        config.run_name = "low_rank_expert"

        with open(save_path + "/config.json", "w") as f:
            json.dump(config.__dict__, f)
        log.info(f"Config saved to {save_path}/config.json")
    log.info("Done")
