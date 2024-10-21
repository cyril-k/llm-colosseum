import sys

from dotenv import load_dotenv
from eval.game import Game, Player1, Player2
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

load_dotenv()


def main():
    # Environment Settings
    game = Game(
        render=True,
        player_1=Player1(
            nickname="Daddy",
            # model="openai:gpt-4o",
            # model="nebius:meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            model="nebius:meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
        ),
        player_2=Player2(
            nickname="Baby",
            # model="openai:gpt-4o-mini",
            # model="nebius:microsoft/Phi-3-mini-4k-instruct-fast",
            model="nebius:mistralai/Mixtral-8x7B-Instruct-v0.1-fast",
        ),
    )
    return game.run()


if __name__ == "__main__":
    main()
