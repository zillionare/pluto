import fire

from pluto.app import start

if __name__ == "__main__":
    fire.Fire({"serve": start})
