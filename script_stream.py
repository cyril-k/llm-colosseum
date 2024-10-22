import sys

from fastapi import FastAPI, Query
from starlette.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time

import cv2
import threading
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from dotenv import load_dotenv
from eval.game import Game, Player1, Player2
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

game_running = False
game = None
game_stopped = False

def get_random_numpy():
    """Generate a random noise frame with 8-bit unsigned integers."""
    return np.random.randint(0, 100, size=(244, 384), dtype=np.uint8)

def make_game(player1_model: str, player2_model: str):
    game = Game(
        render=False,
        player_1=Player1(
            nickname="Daddy",
            model=player1_model,
        ),
        player_2=Player2(
            nickname="Baby",
            model=player2_model,
        ),
    )
    return game

def render_overlay(frame, text):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale back to RGB so we can overlay colored text
    frame_rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    font1 = ImageFont.truetype("PressStart2P-Regular.ttf", 20)
    font2 = ImageFont.truetype("PressStart2P-Regular.ttf", 12)

    line1, line2 = text.split(" | ")

    if "Player 1" in line1:
        text_color = (255, 0, 0)  # Red for Player 1
    elif "Player 2" in line1:
        text_color = (0, 255, 0)  # Green for Player 2
    else:
        text_color = (255, 255, 255)  # Default white

    base_position = (192, 100)     # Adjust the position (x, y) to fit the image properly
    # Get the bounding box for both lines
    bbox_line1 = draw.textbbox((0, 0), line1, font=font1)
    bbox_line2 = draw.textbbox((0, 0), line2, font=font2)
    
    text_width_line1 = bbox_line1[2] - bbox_line1[0]
    text_height_line1 = bbox_line1[3] - bbox_line1[1]
    
    text_width_line2 = bbox_line2[2] - bbox_line2[0]
    text_height_line2 = bbox_line2[3] - bbox_line2[1]

    # Calculate the top-left corner for the first line (centered)
    text_position_line1 = (
        base_position[0] - text_width_line1 // 2,
        base_position[1] - text_height_line1
    )

    # Calculate the top-left corner for the second line (centered and below the first line)
    text_position_line2 = (
        base_position[0] - text_width_line2 // 2,
        base_position[1] + text_height_line1 // 2  # Slightly below the first line
    )

    draw.text(text_position_line1, line1, font=font1, fill=text_color)
    draw.text(text_position_line2, line2, font=font2, fill=text_color)

    frame_with_text = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return cv2.imencode('.jpg', frame_with_text)


async def generate_stream():
    placeholder_path = "./static/placeholder.jpg" 
    last_frame=None
    while True:
        if game and game.frame is not None:
            if game.win_message:
                logger.info("RENDERING WIN MESSAGE")
                ret, jpeg = render_overlay(last_frame, game.win_message)
            else:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                frame_bgr = cv2.cvtColor(game.frame, cv2.COLOR_RGB2BGR)
                ret, jpeg = cv2.imencode('.jpg', frame_bgr, encode_param)
                last_frame = game.frame
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        elif game_running == True:
            noise_frame = get_random_numpy()
            noise_rgb = cv2.cvtColor(noise_frame, cv2.COLOR_GRAY2RGB)
            ret, jpeg = render_overlay(noise_rgb, "STARTING GAME | wait a moment...")
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        await asyncio.sleep(0.1)  # Control frame rate (30 FPS)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def main_page():
    with open("static/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


# def run_game_loop(player1_model: str, player2_model: str):
#     global game_running, game, game_stopped
#     if not game_running:
#         game_running = True
#         game = make_game(player1_model, player2_model)
#         game.run()
#         game_running = False
#         game = None

def run_game_loop(player1_model: str, player2_model: str):
    global game_running, game, game_stopped
    while True:  # Continuous loop to restart the game
        if not game_running:
            game_running = True
            game = make_game(player1_model, player2_model)
            game.run()

            if game.win_message:
                time.sleep(5)  # Pause for 5 seconds to show win_message

            # Reset the game
            logger.info("Restarting the game...")
            game_running = False
            game = None
        else:
            logger.info("Game is already running")
            break


@app.get("/start_game")
async def start_game(player1_model: str = Query(...), player2_model: str = Query(...)):
    global game_running, game_stopped
    if not game_running:
        game_stopped = False
        # Start the game loop in a new thread
        game_thread = threading.Thread(target=run_game_loop, args=(player1_model, player2_model))
        game_thread.daemon = True
        game_thread.start()
        return {"message": "Game started"}
    else:
        return {"message": "Game is already running"}
    
@app.get("/stop_game")
async def stop_game():
    global game_running, game_stopped, game
    if game_running:
        game_stopped = True
        game_running = False
        if game:
            game.stopped = True
        game = None
        return {"message": "Game stopped"}
    return {"message": "Game is not running"}

if __name__ == '__main__':

    # Start the FastAPI server using Uvicorn
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)