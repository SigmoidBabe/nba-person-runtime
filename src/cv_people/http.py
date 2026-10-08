"""Standalone image-upload API; model access is serialized off the event loop."""
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from aiohttp import web


class PeopleDetectionAPI:
    ROUTE = "/api/v1/detect_people"

    def __init__(self, config):
        self.config = config
        self.backend = None
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def register(self, app):
        app.router.add_post(self.ROUTE, self.detect_people)
        app.on_cleanup.append(self.cleanup)

    def process_image(self, image):
        with self.lock:
            if self.backend is None:
                from .backend import PeopleBackend
                self.backend = PeopleBackend(self.config, enable_reid=False)
            detections = self.backend.detect(image, 0.4)
            people = self.backend.convert_detections_to_people(detections)
            return {"people": people, "people_count": len(people)}

    async def detect_people(self, request):
        try:
            if request.content_type.startswith("multipart/"):
                reader = await request.multipart()
                content = None
                async for part in reader:
                    if part.name in ("image", "file"):
                        content = bytearray()
                        while chunk := await part.read_chunk():
                            content.extend(chunk)
                            if len(content) > request.client_max_size:
                                raise web.HTTPRequestEntityTooLarge(
                                    max_size=request.client_max_size, actual_size=len(content))
                        break
                if not content:
                    raise web.HTTPBadRequest(text="An image or file field is required")
            elif request.content_type.startswith("image/") or request.content_type == "application/octet-stream":
                content = await request.read()
            else:
                raise web.HTTPUnsupportedMediaType(text="Send an image body or multipart image/file field")
            if not content:
                raise web.HTTPBadRequest(text="Image is empty")
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise web.HTTPBadRequest(text="Cannot decode image")
            result = await asyncio.get_running_loop().run_in_executor(self.executor, self.process_image, image)
            return web.json_response({"status": 200, "message": "succeeded", "data": result})
        except web.HTTPException:
            raise
        except Exception:
            logging.getLogger(__name__).exception("People detection failed")
            return web.json_response({"status": 500, "message": "people detection failed"}, status=500)

    def _close(self):
        with self.lock:
            if self.backend is not None:
                self.backend.close()
            self.backend = None

    async def cleanup(self, app):
        await asyncio.get_running_loop().run_in_executor(self.executor, self._close)
        self.executor.shutdown(wait=True)
