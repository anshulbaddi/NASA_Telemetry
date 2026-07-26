"""API gateway — FastAPI REST + WebSocket with Kafka fan-out and gRPC proxying."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set

import grpclib.client
from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import uvicorn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generated.telemetry.v1.telemetry_pb2 import SensorData
from generated.pathplanner.v1.path_planner_pb2 import (
    PathPlannerStub, PathRequest, Position, Obstacle as PathObstacle,
)
from generated.fdir.v1.fdir_pb2 import FDIRServiceStub, FDIRStatusRequest
from generated.motorcontroller.v1.motor_controller_pb2 import (
    MotorControllerStub, DriveCommand,
)

# ── config ────────────────────────────────────────────────────────────────────
HTTP_PORT          = int(os.getenv("HTTP_PORT",           "8080"))
KAFKA_BROKERS      = os.getenv("KAFKA_BROKERS",           "localhost:9092")
KAFKA_TOPIC        = os.getenv("KAFKA_TOPIC",              "rover.telemetry")
PATH_PLANNER_ADDR  = os.getenv("PATH_PLANNER_ADDR",       "localhost:50054")
FDIR_ADDR          = os.getenv("FDIR_ADDR",               "localhost:50053")
MOTOR_ADDR         = os.getenv("MOTOR_CONTROLLER_ADDR",   "localhost:50052")
OTLP_ENDPOINT      = os.getenv("OTLP_ENDPOINT",           "http://localhost:4317")
WS_MAX_SUBS        = int(os.getenv("WS_MAX_SUBS",          "500"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── metrics ───────────────────────────────────────────────────────────────────
http_requests  = Counter("gateway_http_requests_total",     "HTTP requests",           ["method", "path", "status"])
ws_connections = Gauge("gateway_ws_connections_active",     "Active WebSocket connections")
frames_fanned  = Counter("gateway_frames_fanned_total",     "Telemetry frames fan-out")
grpc_calls_gw  = Counter("gateway_grpc_calls_total",        "gRPC proxy calls",        ["service", "status"])
req_latency    = Histogram("gateway_request_latency_s",     "HTTP request latency",
                            buckets=[.005, .01, .025, .05, .1, .25, .5, 1.0])

# ── OTel ──────────────────────────────────────────────────────────────────────
def setup_otel() -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("rover.api-gateway")


# ── gRPC stub helpers ──────────────────────────────────────────────────────────
def _host_port(addr: str):
    host, port = addr.rsplit(":", 1)
    return host, int(port)


# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self) -> None:
        # rover_id → set of connected websockets (None key = subscribe all)
        self._subs: Dict[Optional[str], Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, rover_id: Optional[str]) -> bool:
        total = sum(len(v) for v in self._subs.values())
        if total >= WS_MAX_SUBS:
            await ws.close(code=1008)
            return False
        await ws.accept()
        async with self._lock:
            self._subs.setdefault(rover_id, set()).add(ws)
        ws_connections.inc()
        return True

    async def disconnect(self, ws: WebSocket, rover_id: Optional[str]) -> None:
        async with self._lock:
            self._subs.get(rover_id, set()).discard(ws)
        ws_connections.dec()

    async def broadcast(self, rover_id: str, payload: dict) -> None:
        msg = json.dumps(payload)
        dead: List[tuple] = []
        for key in (rover_id, None):
            for ws in list(self._subs.get(key, set())):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append((ws, key))
        for ws, key in dead:
            await self.disconnect(ws, key)


# ── in-memory rover state cache ───────────────────────────────────────────────
_rover_cache: Dict[str, dict] = {}
_manager = ConnectionManager()
_tracer: trace.Tracer


def _sensor_to_dict(frame: SensorData) -> dict:
    return {
        "rover_id":           frame.rover_id,
        "timestamp_ms":       frame.timestamp_ms,
        "latitude":           frame.latitude,
        "longitude":          frame.longitude,
        "altitude_m":         frame.altitude_m,
        "battery_percent":    frame.battery_percent,
        "cpu_temp_celsius":   frame.cpu_temp_celsius,
        "motor_temp_celsius": frame.motor_temp_celsius,
        "motor_load_left":    frame.motor_load_left,
        "motor_load_right":   frame.motor_load_right,
        "velocity_ms":        frame.velocity_ms,
        "heading_degrees":    frame.heading_degrees,
        "status":             str(frame.status),
        "error_code":         frame.error_code,
        "sequence_number":    frame.sequence_number,
    }


# ── Kafka fan-out task ────────────────────────────────────────────────────────
async def kafka_fan_out() -> None:
    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers  = KAFKA_BROKERS,
        group_id           = "api-gateway",
        auto_offset_reset  = "latest",
        enable_auto_commit = True,
        max_poll_records   = 200,
    )
    await consumer.start()
    log.info("API gateway Kafka consumer started")
    try:
        async for msg in consumer:
            try:
                frame = SensorData().parse(msg.value)
            except Exception:
                continue
            d = _sensor_to_dict(frame)
            _rover_cache[frame.rover_id] = d
            frames_fanned.inc()
            await _manager.broadcast(frame.rover_id, d)
    finally:
        await consumer.stop()


# ── FastAPI app ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tracer
    _tracer = setup_otel()
    fan_task = asyncio.create_task(kafka_fan_out())
    yield
    fan_task.cancel()
    try:
        await fan_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title       = "Rover Fleet API Gateway",
    description = "REST + WebSocket gateway for the NASA autonomous rover fleet",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# mount Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())

FastAPIInstrumentor.instrument_app(app)


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp_ms": int(time.time() * 1000)}


@app.get("/rovers")
async def list_rovers():
    return {"rovers": list(_rover_cache.keys()), "count": len(_rover_cache)}


@app.get("/rovers/{rover_id}")
async def get_rover(rover_id: str):
    state = _rover_cache.get(rover_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Rover {rover_id!r} not found")
    return state


@app.get("/rovers/{rover_id}/fdir")
async def get_fdir_status(rover_id: str):
    host, port = _host_port(FDIR_ADDR)
    channel    = grpclib.client.Channel(host=host, port=port)
    stub       = FDIRServiceStub(channel)
    try:
        t0  = time.perf_counter()
        res = await stub.get_status(FDIRStatusRequest(rover_id=rover_id))
        grpc_calls_gw.labels(service="fdir", status="ok").inc()
        return {
            "rover_id":              res.rover_id,
            "current_severity":      str(res.current_severity),
            "active_faults":         list(res.active_faults),
            "last_fault_ms":         res.last_fault_ms,
            "last_recovery_ms":      res.last_recovery_ms,
            "total_anomalies":       res.total_anomalies,
            "anomaly_rate_per_min":  res.anomaly_rate_per_min,
        }
    except Exception as exc:
        grpc_calls_gw.labels(service="fdir", status="error").inc()
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        channel.close()


@app.post("/rovers/{rover_id}/path")
async def plan_path(
    rover_id:   str,
    target_lat: float = Query(...),
    target_lon: float = Query(...),
    max_speed:  float = Query(default=2.0),
    algorithm:  str   = Query(default="astar"),
):
    cached = _rover_cache.get(rover_id)
    if not cached:
        raise HTTPException(status_code=404, detail=f"No telemetry for rover {rover_id!r}")

    host, port = _host_port(PATH_PLANNER_ADDR)
    channel    = grpclib.client.Channel(host=host, port=port)
    stub       = PathPlannerStub(channel)
    try:
        req = PathRequest(
            rover_id    = rover_id,
            current_pos = Position(
                latitude  = cached["latitude"],
                longitude = cached["longitude"],
                altitude  = cached["altitude_m"],
            ),
            target_pos  = Position(latitude=target_lat, longitude=target_lon, altitude=0.0),
            obstacles   = [],
            max_speed   = max_speed,
            algorithm   = algorithm,
        )
        resp = await stub.plan_path(req)
        grpc_calls_gw.labels(service="path-planner", status="ok").inc()
        return {
            "rover_id":     resp.rover_id,
            "path_found":   resp.path_found,
            "total_dist_m": resp.total_dist_m,
            "eta_ms":       resp.eta_ms,
            "waypoints":    [
                {
                    "lat":  wp.position.latitude,
                    "lon":  wp.position.longitude,
                    "alt":  wp.position.altitude,
                    "speed": wp.speed_ms,
                    "eta":  wp.eta_ms,
                    "checkpoint": wp.is_checkpoint,
                }
                for wp in resp.waypoints
            ],
            "reason": resp.reason,
        }
    except Exception as exc:
        grpc_calls_gw.labels(service="path-planner", status="error").inc()
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        channel.close()


@app.post("/rovers/{rover_id}/estop")
async def emergency_stop(rover_id: str):
    host, port = _host_port(MOTOR_ADDR)
    channel    = grpclib.client.Channel(host=host, port=port)
    stub       = MotorControllerStub(channel)
    try:
        resp = await stub.emergency_stop(DriveCommand(
            rover_id       = rover_id,
            emergency_stop = True,
        ))
        grpc_calls_gw.labels(service="motor-controller", status="ok").inc()
        return {
            "rover_id":      resp.rover_id,
            "command_acked": resp.command_acked,
            "reason":        resp.reason,
            "timestamp_ms":  resp.timestamp_ms,
        }
    except Exception as exc:
        grpc_calls_gw.labels(service="motor-controller", status="error").inc()
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        channel.close()


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws/telemetry")
async def ws_all_rovers(websocket: WebSocket):
    """Subscribe to all rovers' telemetry."""
    if not await _manager.connect(websocket, rover_id=None):
        return
    try:
        while True:
            await websocket.receive_text()  # keep alive / ping
    except WebSocketDisconnect:
        pass
    finally:
        await _manager.disconnect(websocket, rover_id=None)


@app.websocket("/ws/telemetry/{rover_id}")
async def ws_single_rover(websocket: WebSocket, rover_id: str):
    """Subscribe to a single rover's telemetry stream."""
    if not await _manager.connect(websocket, rover_id=rover_id):
        return
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await _manager.disconnect(websocket, rover_id=rover_id)


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = HTTP_PORT,
        workers = 1,
        log_level = "info",
    )
