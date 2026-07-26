"""Path-planner gRPC service — A* navigation with dynamic obstacle avoidance."""
from __future__ import annotations

import asyncio
import heapq
import logging
import math
import os
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

import grpclib.server
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, Gauge, start_http_server

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generated.pathplanner.v1.path_planner_pb2 import (
    PathPlannerBase,
    PathRequest, PathResponse,
    PathAdjustment, PositionUpdate,
    Position, Waypoint, Obstacle,
)

# ── config ────────────────────────────────────────────────────────────────────
GRPC_PORT    = int(os.getenv("GRPC_PORT",    "50054"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "9092"))
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT",  "http://localhost:4317")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── metrics ───────────────────────────────────────────────────────────────────
paths_planned   = Counter("pathplanner_paths_planned_total",     "Paths computed",         ["algorithm", "found"])
planning_time   = Histogram("pathplanner_planning_seconds",      "A* computation time",
                             buckets=[.001, .005, .01, .05, .1, .5, 1.0])
waypoints_gauge = Gauge("pathplanner_last_waypoints",            "Waypoints in last path")

# ── OTel ──────────────────────────────────────────────────────────────────────
def setup_otel() -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("rover.path-planner")


# ── A* planner ────────────────────────────────────────────────────────────────
_GRID_RES  = 0.00005   # ~5 m per cell in degrees
_MAX_ITER  = 50_000


def _lat_lon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
    return round(lat / _GRID_RES), round(lon / _GRID_RES)


def _grid_to_pos(r: int, c: int) -> Tuple[float, float]:
    return r * _GRID_RES, c * _GRID_RES


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _is_blocked(r: int, c: int, blocked: set) -> bool:
    return (r, c) in blocked


def _build_blocked_set(obstacles: List[Obstacle]) -> set:
    blocked = set()
    for obs in obstacles:
        cr, cc = _lat_lon_to_grid(obs.center.latitude, obs.center.longitude)
        cells  = math.ceil(obs.radius_m / (_GRID_RES * 111_320)) + 2  # padding
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                if math.hypot(dr, dc) <= cells:
                    blocked.add((cr + dr, cc + dc))
    return blocked


def astar(
    start: Position,
    goal:  Position,
    obstacles: List[Obstacle],
    max_speed: float,
) -> Optional[List[Waypoint]]:
    blocked   = _build_blocked_set(obstacles)
    sr, sc    = _lat_lon_to_grid(start.latitude,  start.longitude)
    gr, gc    = _lat_lon_to_grid(goal.latitude,   goal.longitude)

    if _is_blocked(gr, gc, blocked):
        return None

    def h(r: int, c: int) -> float:
        lat, lon = _grid_to_pos(r, c)
        glat, glon = _grid_to_pos(gr, gc)
        return _haversine_m(lat, lon, glat, glon)

    heap  = [(h(sr, sc), 0.0, sr, sc)]
    came_from: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {(sr, sc): None}
    g_score: Dict[Tuple[int,int], float] = {(sr, sc): 0.0}
    iters = 0

    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    step_cost = [1.0, 1.0, 1.0, 1.0, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]

    while heap and iters < _MAX_ITER:
        _, g, r, c = heapq.heappop(heap)
        iters += 1

        if r == gr and c == gc:
            # reconstruct path
            path: List[Tuple[int,int]] = []
            cur: Optional[Tuple[int,int]] = (r, c)
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()

            waypoints = []
            now_ms = int(time.time() * 1000)
            total_dist = 0.0
            prev_lat, prev_lon = start.latitude, start.longitude

            for i, (wr, wc) in enumerate(path[::3] or path):  # thin to ~1/3
                wlat, wlon = _grid_to_pos(wr, wc)
                seg_dist   = _haversine_m(prev_lat, prev_lon, wlat, wlon)
                total_dist += seg_dist
                eta_ms     = now_ms + int(total_dist / max(max_speed, 0.1) * 1000)
                waypoints.append(Waypoint(
                    position      = Position(latitude=wlat, longitude=wlon, altitude=start.altitude),
                    heading       = 0.0,
                    speed_ms      = max_speed,
                    eta_ms        = eta_ms,
                    is_checkpoint = (i % 5 == 0),
                ))
                prev_lat, prev_lon = wlat, wlon

            return waypoints

        if g > g_score.get((r, c), float("inf")) + 1e-9:
            continue

        for (dr, dc), cost in zip(dirs, step_cost):
            nr, nc = r + dr, c + dc
            if _is_blocked(nr, nc, blocked):
                continue
            ng = g + cost * _GRID_RES * 111_320  # metres
            if ng < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)]  = ng
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(heap, (ng + h(nr, nc), ng, nr, nc))

    return None  # no path found or max iterations exceeded


# ── gRPC service implementation ───────────────────────────────────────────────
_tracer: trace.Tracer


class PathPlannerService(PathPlannerBase):

    async def plan_path(self, path_request: PathRequest) -> PathResponse:
        with _tracer.start_as_current_span("pathplanner.plan_path") as span:
            span.set_attribute("rover.id",  path_request.rover_id)
            span.set_attribute("algorithm", path_request.algorithm or "astar")

            t0 = time.perf_counter()
            waypoints = astar(
                path_request.current_pos,
                path_request.target_pos,
                path_request.obstacles,
                path_request.max_speed or 2.0,
            )
            elapsed = time.perf_counter() - t0
            planning_time.observe(elapsed)

            found = waypoints is not None
            paths_planned.labels(algorithm=path_request.algorithm or "astar", found=str(found)).inc()

            if found:
                waypoints_gauge.set(len(waypoints))
                total_dist = sum(
                    _haversine_m(
                        waypoints[i].position.latitude,  waypoints[i].position.longitude,
                        waypoints[i+1].position.latitude, waypoints[i+1].position.longitude,
                    )
                    for i in range(len(waypoints) - 1)
                )
                eta_ms = waypoints[-1].eta_ms if waypoints else 0
                span.set_attribute("path.waypoints", len(waypoints))
                span.set_attribute("path.dist_m",    total_dist)
                return PathResponse(
                    rover_id     = path_request.rover_id,
                    waypoints    = waypoints,
                    total_dist_m = total_dist,
                    eta_ms       = eta_ms,
                    path_found   = True,
                    reason       = f"A* found path in {elapsed*1000:.1f}ms",
                )
            else:
                return PathResponse(
                    rover_id   = path_request.rover_id,
                    path_found = False,
                    reason     = "No feasible path (all routes blocked or max iterations exceeded)",
                )

    async def stream_path(self, path_request: PathRequest) -> AsyncIterator[Waypoint]:
        response = await self.plan_path(path_request)
        for wp in response.waypoints:
            await asyncio.sleep(0.05)  # pace delivery
            yield wp

    async def update_position(
        self, request: AsyncIterator[PositionUpdate]
    ) -> AsyncIterator[PathAdjustment]:
        async for update in request:
            # Simplified reactive reroute: compute mini-path from current to a
            # slight look-ahead point when position drifts significantly
            look_ahead = Position(
                latitude  = update.current_pos.latitude  + 0.001,
                longitude = update.current_pos.longitude + 0.001,
                altitude  = update.current_pos.altitude,
            )
            wps = astar(update.current_pos, look_ahead, [], update.velocity or 2.0)
            yield PathAdjustment(
                rover_id  = update.rover_id,
                waypoints = wps[:5] if wps else [],
                rerouted  = wps is not None,
            )


# ── entrypoint ────────────────────────────────────────────────────────────────
async def main() -> None:
    global _tracer
    start_http_server(METRICS_PORT)
    _tracer = setup_otel()

    server = grpclib.server.Server([PathPlannerService()])
    await server.start("0.0.0.0", GRPC_PORT)
    log.info("PathPlanner gRPC server listening on :%d", GRPC_PORT)
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
