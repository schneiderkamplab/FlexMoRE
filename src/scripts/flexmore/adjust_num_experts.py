import json
import os
import typer

def main(
    num_experts: int,
    paths: list[str],
):
    for path in paths:
        c = os.path.join(path, "config.json")
        d = json.load(open(c, "rt"))
        o = d["num_experts_per_tok"]
        d["num_experts_per_tok"] = num_experts
        open(c, "wt").write(json.dumps(d, indent=2))
        print(f"Adjusted {path} from {o} to {num_experts} active experts out of {d['num_experts']}")

if __name__ == "__main__":
    typer.run(main)
