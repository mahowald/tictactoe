import click
from .fit import fit
from .play import play
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

@click.command()
@click.option("--mode", default="play", help="fit or play")
def main(mode="play"):
    if mode == "fit":
        res = fit()
        sys.stdout.buffer.write(res)
        sys.stdout.flush()
    elif mode == "play":
        play()

if __name__ == "__main__":
    main()