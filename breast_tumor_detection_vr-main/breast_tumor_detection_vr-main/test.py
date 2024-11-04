import asyncio
import websockets
import cv2
import base64
import os

video_files = {
    "/video_dl_classification": 'video_dl_classification.mp4',
    "/video_dl_segmentation": 'video_dl_segmentation.mp4',
    "/video_llm_segmentation": 'video_llm_segmentation.mp4',
    "/video_llm_classification": 'video_llm_classification.mp4',
    "/input_video_dl": 'input_video_dl.mp4',
    "/input_video_llm": 'input_video_llm.mp4'
}

image_files = {
    "/input_image_dl": 'input_image_dl.png',
    "/input_image_llm": 'input_image_llm.png',
    "/image_dl_classification": 'image_dl_classification.png',
    "/image_llm_classification": 'image_llm_classification.png',
    "/image_dl_segmentation": 'image_dl_segmentation.png',
    "/image_llm_segmentation": 'image_llm_segmentation.png'
}

connected_clients = set()

async def send_frames(websocket, path):
    if path not in video_files:
        return

    current_video_file = video_files[path]

    try:
        connected_clients.add(websocket)

        while not os.path.exists(current_video_file):
            await asyncio.sleep(1)  # Wait for the video file to exist

        cap = cv2.VideoCapture(current_video_file)  # Open the video file

        while websocket in connected_clients:
            if not cap.isOpened():
                cap = cv2.VideoCapture(current_video_file)  # Restart the video file

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart from the first frame
                continue

            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            try:
                await websocket.send(frame_data)
                print("Frame sent")
            except websockets.exceptions.ConnectionClosed:
                break

            await asyncio.sleep(1 / 30)  # Adjust to match the frame rate of your video

        cap.release()

    finally:
        connected_clients.remove(websocket)

async def send_image(websocket, path):
    if path not in image_files:
        return

    current_image_file = image_files[path]

    try:
        connected_clients.add(websocket)

        while not os.path.exists(current_image_file):
            await asyncio.sleep(1)  # Wait for the image file to exist

        while websocket in connected_clients:
            image = cv2.imread(current_image_file)  # Open the image file

            _, buffer = cv2.imencode('.jpg', image)
            image_data = base64.b64encode(buffer).decode('utf-8')

            try:
                await websocket.send(image_data)
                print("Image sent")
            except websockets.exceptions.ConnectionClosed:
                break

            await asyncio.sleep(1)  # Send image every second

    finally:
        connected_clients.remove(websocket)

async def handler(websocket, path):
    if path in video_files:
        await send_frames(websocket, path)
    elif path in image_files:
        await send_image(websocket, path)

async def main():
    async with websockets.serve(handler, 'localhost', 8766):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
