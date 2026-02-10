from transformers import AutoTokenizer
import typer

def main(
    target: str,
    source: str = "allenai/Flex-code-2x7B-1T"
):
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.save_pretrained(target)

if __name__ == "__main__":\
    typer.run(main)
