#!/usr/bin/env python
from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from crocodile import Crocodile

load_dotenv()


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(Crocodile, "croco")
    parser.add_argument(
        "--croco.mongo-uri",
        type=str,
        help="MongoDB connection URI",
        default="mongodb://mongodb:27017",
    )
    args = parser.parse_args()

    print("🚀 Starting the entity linking process...")
    croco = Crocodile(**args.croco)
    croco.run()
    print("✅ Entity linking process completed.")


if __name__ == "__main__":
    main()
