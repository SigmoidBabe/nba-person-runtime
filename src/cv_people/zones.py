"""Zone geometry adapted from BaseRuntimeSession; no transport or publishing."""
import json
import cv2
import numpy as np
from shapely.geometry import Polygon

class ZoneAnalytics:
    def __init__(self, spec):
        self.zones = {}
        for index, zone in enumerate(spec.get("zones", [])):
            name = str(zone.get("zone_name") or zone.get("camera_zone_id") or zone.get("zone_id") or index)
            if name in self.zones:
                raise ValueError(f"Duplicate zone name: {name}")
            self.zones[name] = {
                **zone,
                "polygon": zone.get("points", zone.get("polygon", [])),
                "zone_type": zone.get("type", zone.get("zone_type", "polygon")),
            }
        self.zone_polygons = {}
        self.zone_lines = {}
        self.zone_meta = {}
        self.zone_frame_shape = None
        self.zone_min_overlap = float(spec["config"].get("zone_min_overlap", 0.3))

    def normalize_point(self, point, frame_width, frame_height):
        if isinstance(point, dict):
            point = (point.get("x"), point.get("y"))

        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None

        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None

        if 0 <= x <= 1 and 0 <= y <= 1:
            x *= frame_width
            y *= frame_height

        return int(x), int(y)

    def polygon_points(self, polygon):
        if isinstance(polygon, str):
            try:
                polygon = json.loads(polygon)
            except ValueError:
                return []

        if isinstance(polygon, dict):
            polygon = polygon.get("points") or polygon.get("polygon") or []

        while (
            isinstance(polygon, (list, tuple))
            and len(polygon) == 1
            and isinstance(polygon[0], (list, tuple))
        ):
            polygon = polygon[0]

        if not isinstance(polygon, (list, tuple)):
            return []

        return polygon

    def normalize_polygon_points(self, polygon, frame_width, frame_height):
        points = [
            self.normalize_point(point, frame_width, frame_height)
            for point in self.polygon_points(polygon)
        ]
        return [point for point in points if point is not None]

    def build_zone_polygons(self, frame):
        frame_height, frame_width = frame.shape[:2]
        zone_polygons = {}
        zone_lines = {}
        zone_meta = {}

        for zone_name, zone_data in self.zones.items():
            if not isinstance(zone_data, dict):
                continue

            polygon = zone_data.get("polygon")
            if not polygon:
                continue

            points = self.normalize_polygon_points(
                polygon,
                frame_width,
                frame_height,
            )
            zone_key = str(zone_name)
            zone_type = str(zone_data.get("zone_type") or "").strip().lower()
            if zone_type == "line" or len(points) == 2:
                if len(points) != 2 or points[0] == points[1]:
                    continue
                zone_lines[zone_key] = tuple(points)
                zone_meta[zone_key] = {
                    key: value
                    for key, value in zone_data.items()
                    if key != "polygon"
                }
                continue

            if len(points) < 3:
                continue

            poly = Polygon(points)
            if poly.is_empty or not poly.is_valid or poly.area <= 0:
                continue

            zone_polygons[zone_key] = poly
            zone_meta[zone_key] = {
                key: value
                for key, value in zone_data.items()
                if key != "polygon"
            }

        self.zone_polygons = zone_polygons
        self.zone_lines = zone_lines
        self.zone_meta = zone_meta
        self.zone_frame_shape = (frame_height, frame_width)

    def ensure_zone_polygons(self, frame):
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return

        frame_shape = frame.shape[:2]
        if self.zone_frame_shape == frame_shape:
            return

        self.build_zone_polygons(frame)

    def bbox_to_xyxy(self, box, box_format="xyxy"):
        if box is None:
            return None

        try:
            values = [float(value) for value in box[:4]]
        except (TypeError, ValueError):
            return None

        if len(values) < 4:
            return None

        if box_format == "tlwh":
            x, y, width, height = values
            return x, y, x + width, y + height

        return values[0], values[1], values[2], values[3]

    def box_polygon(self, box, box_format="xyxy"):
        xyxy = self.bbox_to_xyxy(box, box_format=box_format)
        if xyxy is None:
            return None

        x1, y1, x2, y2 = xyxy
        if x2 <= x1 or y2 <= y1:
            return None

        return Polygon([
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ])

    def zones_for_box(self, box, box_format="xyxy", min_overlap=None):
        bbox_polygon = self.box_polygon(box, box_format=box_format)
        if bbox_polygon is None or bbox_polygon.area <= 0:
            return []

        if min_overlap is None:
            min_overlap = self.zone_min_overlap

        zone_names = []
        for zone_name, zone_polygon in self.zone_polygons.items():
            overlap = bbox_polygon.intersection(zone_polygon).area / bbox_polygon.area
            if overlap >= min_overlap and bbox_polygon.intersects(zone_polygon):
                zone_names.append(zone_name)

        return zone_names

    def count_people_in_zones(self, people, box_key="box", box_format="xyxy"):
        zone_counts = {zone_name: 0 for zone_name in self.zone_polygons}
        enriched_people = []

        for index, person in enumerate(people, start=1):
            person_data = dict(person)
            person_zones = self.zones_for_box(
                person_data.get(box_key),
                box_format=box_format,
            )
            person_data["zones"] = person_zones
            person_data.setdefault("person_index", index)

            for zone_name in person_zones:
                zone_counts[zone_name] += 1

            enriched_people.append(person_data)

        return zone_counts, enriched_people

    def draw_zones(self, frame):
        frame_height, frame_width = frame.shape[:2]
        for zone_name, zone_data in self.zones.items():
            polygon = zone_data.get("polygon")
            if not polygon:
                continue

            points = self.normalize_polygon_points(
                polygon,
                frame_width,
                frame_height,
            )
            if len(points) < 2:
                continue

            points = np.array(points, dtype=np.int32)
            cv2.polylines(
                frame,
                [points],
                len(points) >= 3,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                str(zone_name),
                tuple(points[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

