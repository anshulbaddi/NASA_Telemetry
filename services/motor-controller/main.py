"""Motor-controller gRPC service — differential-drive model with fault detection."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import AsyncIterator, Dict

import grpclib.server
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, Gauge, start_http_server

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generated.motorcontroller.v1.motor_controller_pb2 import (
    MotorControllerBase,
    DriveCommand, MotorResponse,
    TelemetryRequest, MotorState,
    MotorFault,
)

# ── config ────────────────────────────────────────────────────────────────────
GRPC_PORT     = int(os.getenv("GRPC_PORT",    "50052"))
METRICS_PORT  = int(os.getenv("METRICS_PORT", "9094"))
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT",   "http://localhost:4317")

MAX_SPEED_MS        = float(os.getenv("MAX_SPEED_MS",       "5.0"))
MAX_TURN_RATE       = float(os.getenv("MAX_TURN_RATE",       "1.0"))  # rad/s
OVERTEMP_CELSIUS    = float(os.getenv("OVERTEMP_CELSIUS",   "85.0"))
OVERLOAD_THRESHOLD  = float(os.getenv("OVERLOAD_THRESHOLD",  "0.95"))
WHEEL_BASE_M        = float(os.getenv("WHEEL_BASE_M",        "0.5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── metrics ───────────────────────────────────────────────────────────────────
commands_total   = Counter("motor_commands_total",       "Drive commands received",      ["acked"])
estops_total     = Counter("motor_estops_total",         "Emergency stops triggered")
faults_total     = Counter("motor_faults_total",         "Motor fault events",           ["side", "fault"])
command_latency  = Histogram("motor_command_latency_s",  "Command processing latency",
                              buckets=[.0001, .0005, .001, .005, .01])
left_temp_gauge  = Gauge("motor_left_temp_celsius",      "Left motor temperature")
right_temp_gauge = Gauge("motor_right_temp_celsius",     "Right motor temperature")

# ── OTel ──────────────────────────────────────────────────────────────────────
def setup_otel() -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("rover.motor-controller")


# ── differential-drive model ──────────────────────────────────────────────────
class MotorModel:
    """Per-rover motor state with first-order dynamics."""

    def __init__(self, rover_id: str) -> None:
        self.rover_id      = rover_id
        self.speed_left    = 0.0
        self.speed_right   = 0.0
        self.temp_left     = 25.0
        self.temp_right    = 25.0
        self.load_left     = 0.0
        self.load_right    = 0.0
        self.rpm_left      = 0.0
        self.rpm_right     = 0.0
        self.fault_left    = MotorFault.MOTOR_FAULT_NONE
        self.fault_right   = MotorFault.MOTOR_FAULT_NONE
        self.current_left  = 0.0
        self.current_right = 0.0
        self._stopped      = False

    def apply_command(self, cmd: DriveCommand) -> None:
        if cmd.emergency_stop or self._stopped:
            self._stopped   = True
            self.speed_left = self.speed_right = 0.0
            return

        v = max(-MAX_SPEED_MS, min(MAX_SPEED_MS, cmd.target_speed_ms))
        w = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, cmd.turn_rate))

        # differential drive kinematics
        v_left  = v - w * WHEEL_BASE_M / 2
        v_right = v + w * WHEEL_BASE_M / 2

        # first-order lag τ ≈ 0.1 s
        alpha = 0.85
        self.speed_left  = alpha * self.speed_left  + (1 - alpha) * v_left
        self.speed_right = alpha * self.speed_right + (1 - alpha) * v_right

    def _tick(self) -> None:
        """Simulate thermal/electrical dynamics."""
        for side in ("left", "right"):
            speed = abs(getattr(self, f"speed_{side}"))
            load  = speed / MAX_SPEED_MS * 0.8 + 0.05
            setattr(self, f"load_{side}", min(1.0, load))

            # thermal model: power input heats, ambient cools
            temp = getattr(self, f"temp_{side}")
            power_input = load ** 2 * 50.0
            cooling     = (temp - 25.0) * 0.5
            temp += (power_input - cooling) * 0.005
            setattr(self, f"temp_{side}", min(temp, 150.0))

            # RPM ≈ 100 rpm / (m/s)
            rpm = speed * 100.0
            setattr(self, f"rpm_{side}", rpm)

            # current amps: P = I²R, ~0.5 Ω winding
            current = math.sqrt(max(0, load * 50.0 / 0.5))
            setattr(self, f"current_{side}", current)

            # fault detection
            temp = getattr(self, f"temp_{side}")
            load_val = getattr(self, f"load_{side}")
            if temp >= OVERTEMP_CELSIUS:
                fault = MotorFault.MOTOR_FAULT_OVERTEMP
            elif load_val >= OVERLOAD_THRESHOLD:
                fault = MotorFault.MOTOR_FAULT_OVERLOAD
            elif speed > 0 and rpm < 5:
                fault = MotorFault.MOTOR_FAULT_STALL
            else:
                fault = MotorFault.MOTOR_FAULT_NONE
            setattr(self, f"fault_{side}", fault)

    def states(self) -> tuple:
        self._tick()
        left_temp_gauge.set(self.temp_left)
        right_temp_gauge.set(self.temp_right)

        left = MotorState(
            speed_ms     = self.speed_left,
            load         = self.load_left,
            temperature  = self.temp_left,
            fault        = self.fault_left,
            rpm          = self.rpm_left,
            current_amps = self.current_left,
        )
        right = MotorState(
            speed_ms     = self.speed_right,
            load         = self.load_right,
            temperature  = self.temp_right,
            fault        = self.fault_right,
            rpm          = self.rpm_right,
            current_amps = self.current_right,
        )

        for side, state in (("left", left), ("right", right)):
            if state.fault != MotorFault.MOTOR_FAULT_NONE:
                faults_total.labels(side=side, fault=str(state.fault)).inc()

        return left, right


# ── gRPC service implementation ───────────────────────────────────────────────
_tracer: trace.Tracer
_models: Dict[str, MotorModel] = {}


def _get_model(rover_id: str) -> MotorModel:
    if rover_id not in _models:
        _models[rover_id] = MotorModel(rover_id)
    return _models[rover_id]


class MotorControllerService(MotorControllerBase):

    async def send_command(self, drive_command: DriveCommand) -> MotorResponse:
        with _tracer.start_as_current_span("motor.send_command") as span:
            span.set_attribute("rover.id",    drive_command.rover_id)
            span.set_attribute("target.speed", drive_command.target_speed_ms)
            span.set_attribute("turn_rate",    drive_command.turn_rate)

            t0    = time.perf_counter()
            model = _get_model(drive_command.rover_id)
            model.apply_command(drive_command)
            left, right = model.states()
            elapsed = time.perf_counter() - t0
            command_latency.observe(elapsed)

            acked = (left.fault == MotorFault.MOTOR_FAULT_NONE and
                     right.fault == MotorFault.MOTOR_FAULT_NONE)
            commands_total.labels(acked=str(acked)).inc()

            reason = "OK" if acked else f"Fault L={left.fault} R={right.fault}"
            return MotorResponse(
                rover_id      = drive_command.rover_id,
                left_motor    = left,
                right_motor   = right,
                command_acked = acked,
                reason        = reason,
                timestamp_ms  = int(time.time() * 1000),
            )

    async def emergency_stop(self, drive_command: DriveCommand) -> MotorResponse:
        estops_total.inc()
        with _tracer.start_as_current_span("motor.emergency_stop") as span:
            span.set_attribute("rover.id", drive_command.rover_id)
            model = _get_model(drive_command.rover_id)
            stop  = DriveCommand(
                rover_id        = drive_command.rover_id,
                emergency_stop  = True,
                target_speed_ms = 0.0,
            )
            model.apply_command(stop)
            left, right = model.states()
            log.warning("EMERGENCY STOP rover=%s", drive_command.rover_id)
            return MotorResponse(
                rover_id      = drive_command.rover_id,
                left_motor    = left,
                right_motor   = right,
                command_acked = True,
                reason        = "Emergency stop applied",
                timestamp_ms  = int(time.time() * 1000),
            )

    async def stream_telemetry(
        self, telemetry_request: TelemetryRequest
    ) -> AsyncIterator[MotorResponse]:
        rover_id    = telemetry_request.rover_id
        interval_ms = max(100, telemetry_request.interval_ms or 500)
        model       = _get_model(rover_id)

        while True:
            await asyncio.sleep(interval_ms / 1000.0)
            left, right = model.states()
            yield MotorResponse(
                rover_id      = rover_id,
                left_motor    = left,
                right_motor   = right,
                command_acked = True,
                reason        = "telemetry",
                timestamp_ms  = int(time.time() * 1000),
            )


# ── entrypoint ────────────────────────────────────────────────────────────────
async def main() -> None:
    global _tracer
    start_http_server(METRICS_PORT)
    _tracer = setup_otel()

    server = grpclib.server.Server([MotorControllerService()])
    await server.start("0.0.0.0", GRPC_PORT)
    log.info("MotorController gRPC server listening on :%d", GRPC_PORT)
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
