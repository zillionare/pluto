"""Console script for pluto."""

import fire


def help():
    print("pluto")
    print("=" * len("pluto"))
    print("Skeleton project created by Python Project Wizard (ppw)")


def main():
    fire.Fire({"help": help})


if __name__ == "__main__":
    main()  # pragma: no cover
