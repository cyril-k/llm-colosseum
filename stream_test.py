from fastapi import FastAPI
from starlette.responses import StreamingResponse
import asyncio

import diambra.arena
import cv2
import threading

app = FastAPI()

def game_loop():
    global latest_frame
    # Environment creation
    env = diambra.arena.make("sfiii3n", render_mode="rgb_array")

    # Environment reset
    observation, info = env.reset(seed=42)

    # Agent-Environment interaction loop
    while True:
        # Render environment as an RGB array
        frame = env.render()

        # Convert the RGB frame to BGR (for OpenCV compatibility) and then encode it to JPEG
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame_bgr)
        
        # Update the global latest_frame with the new JPEG image
        if ret:
            latest_frame = jpeg.tobytes()

        # Action random sampling
        actions = env.action_space.sample()

        # Environment stepping
        observation, reward, terminated, truncated, info = env.step(actions)

        # Episode end (Done condition) check
        if terminated or truncated:
            observation, info = env.reset()
            break

    # Environment shutdown
    env.close()

async def generate_stream():
    global latest_frame
    while True:
        if latest_frame is not None:
            # Return the latest frame as part of the MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        await asyncio.sleep(0.033)  # Sleep to control the frame rate (~30 FPS)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # Start the game loop in a separate thread
    game_thread = threading.Thread(target=game_loop)
    game_thread.daemon = True
    game_thread.start()

    # Start the FastAPI server using Uvicorn
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)