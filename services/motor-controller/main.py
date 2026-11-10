"""
Motor-controller gRPC service — differential-drive model with fault detection.

Supports two hardware backends selected via MOTOR_BACKEND env var:

  simulated (default):
    Software differential-drive model with simulated thermal/electrical dynamics.
    Use this on development machines and for ground-station simulation.

  gpio:
    Physical motor control via Jetson.GPIO PWM pins.
    Requires a compatible H-bridge motor driver (e.g. L298N, DRV8833).
    Configure pin numbers via MOTOR_LEFT_PWM_PIN etc.

  serial:
    Physical motor control via serial/UART (e.g. Sabertooth, ODrive, RoboClaw).
    Configure port and baud via MOTOR_SERIAL_PORT and MOTOR_SERIAL_BAUD.

The gRPC interface and protobuf messages are identical regardless of backend.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Optional, Tuple

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
MOTOR_BACKEND = os.getenv("MOTOR_BACKEND",    "simulated")   # simulated | gpio | serial

MAX_SPEED_MS        = float(os.getenv("MAX_SPEED_MS",       "5.0"))
MAX_TURN_RATE       = float(os.getenv("MAX_TURN_RATE",       "1.0"))  # rad/s
OVERTEMP_CELSIUS    = float(os.getenv("OVERTEMP_CELSIUS",   "85.0"))
OVERLOAD_THRESHOLD  = float(os.getenv("OVERLOAD_THRESHOLD",  "0.95"))
WHEEL_BASE_M        = float(os.getenv("WHEEL_BASE_M",        "0.5"))

# GPIO backend pin config (Jetson board pin numbers)
GPIO_LEFT_PWM_PIN   = int(os.getenv("MOTOR_LEFT_PWM_PIN",   "33"))
GPIO_LEFT_DIR1_PIN  = int(os.getenv("MOTOR_LEFT_DIR1_PIN",  "35"))
GPIO_LEFT_DIR2_PIN  = int(os.getenv("MOTOR_LEFT_DIR2_PIN",  "37"))
GPIO_RIGHT_PWM_PIN  = int(os.getenv("MOTOR_RIGHT_PWM_PIN",  "32"))
GPIO_RIGHT_DIR1_PIN = int(os.getenv("MOTOR_RIGHT_DIR1_PIN", "36"))
GPIO_RIGHT_DIR2_PIN = int(os.getenv("MOTOR_RIGHT_DIR2_PIN", "38"))
GPIO_PWM_FREQ_HZ    = int(os.getenv("MOTOR_PWM_FREQ_HZ",    "1000"))

# Serial backend config
SERIAL_PORT         = os.getenv("MOTOR_SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUD         = int(os.getenv("MOTOR_SERIAL_BAUD", "9600"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Hardware Abstraction Layer (HAL) ──────────────────────────────────────────

class MotorHAL(ABC):
    """
    Abstract motor hardware interface.

    Implementations translate normalised left/right wheel speeds (m/s, signed)
    into hardware-specific commands.  The gRPC service layer calls apply() and
    read() without knowing which backend is active.
    """

    @abstractmethod
    def apply(self, speed_left_ms: float, speed_right_ms: float) -> None:
        """Apply left and right wheel speeds in m/s (negative = reverse)."""

    @abstractmethod
    def read(self) -> Tuple[float, float]:
        """Return actual (left_speed_ms, right_speed_ms) — may be estimated."""

    @abstractmethod
    def stop(self) -> None:
        """Emergency stop: immediately set both channels to zero."""

    def close(self) -> None:
        """Release hardware resources (GPIO, serial port, etc.)."""


class GPIOMotorHAL(MotorHAL):
    """
    Motor control via Jetson.GPIO PWM.

    Expected wiring per motor channel:
      PWM pin  → motor driver ENABLE / speed input
      DIR1 pin → direction bit 1
      DIR2 pin → direction bit 2

    For L298N: DIR1=HIGH, DIR2=LOW → forward; DIR1=LOW, DIR2=HIGH → reverse.
    """

    def __init__(self) -> None:
        try:
            import Jetson.GPIO as GPIO
            self._GPIO = GPIO
        except ImportError:
            try:
                import RPi.GPIO as GPIO  # fallback for Raspberry Pi dev
                self._GPIO = GPIO
            except ImportError:
                raise ImportError(
                    "MOTOR_BACKEND=gpio requires Jetson.GPIO. "
                    "Install with: sudo pip3 install Jetson.GPIO"
                )

        GPIO = self._GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(GPIO_LEFT_PWM_PIN,   GPIO.OUT)
        GPIO.setup(GPIO_LEFT_DIR1_PIN,  GPIO.OUT)
        GPIO.setup(GPIO_LEFT_DIR2_PIN,  GPIO.OUT)
        GPIO.setup(GPIO_RIGHT_PWM_PIN,  GPIO.OUT)
        GPIO.setup(GPIO_RIGHT_DIR1_PIN, GPIO.OUT)
        GPIO.setup(GPIO_RIGHT_DIR2_PIN, GPIO.OUT)

        self._pwm_left  = GPIO.PWM(GPIO_LEFT_PWM_PIN,  GPIO_PWM_FREQ_HZ)
        self._pwm_right = GPIO.PWM(GPIO_RIGHT_PWM_PIN, GPIO_PWM_FREQ_HZ)
        self._pwm_left.start(0)
        self._pwm_right.start(0)
        self._left_speed  = 0.0
        self._right_speed = 0.0
        log.info("GPIO motor HAL initialised: PWM_FREQ=%d Hz", GPIO_PWM_FREQ_HZ)

    def apply(self, speed_left_ms: float, speed_right_ms: float) -> None:
        self._set_channel(speed_left_ms,  self._pwm_left,
                          GPIO_LEFT_DIR1_PIN,  GPIO_LEFT_DIR2_PIN,  "left")
        self._set_channel(speed_right_ms, self._pwm_right,
                          GPIO_RIGHT_DIR1_PIN, GPIO_RIGHT_DIR2_PIN, "right")
        self._left_speed  = speed_left_ms
        self._right_speed = speed_right_ms

    def _set_channel(self, speed_ms: float, pwm_obj,
                     dir1_pin: int, dir2_pin: int, label: str) -> None:
        GPIO = self._GPIO
        duty  = min(100.0, abs(speed_ms) / MAX_SPEED_MS * 100.0)
        fwd   = speed_ms >= 0
        GPIO.output(dir1_pin, GPIO.HIGH if fwd else GPIO.LOW)
        GPIO.output(dir2_pin, GPIO.LOW  if fwd else GPIO.HIGH)
        pwm_obj.ChangeDutyCycle(duty)

    def read(self) -> Tuple[float, float]:
        return self._left_speed, self._right_speed

    def stop(self) -> None:
        self.apply(0.0, 0.0)

    def close(self) -> None:
        self.stop()
        self._pwm_left.stop()
        self._pwm_right.stop()
        self._GPIO.cleanup()
        log.info("GPIO motor HAL closed")


class SerialMotorHAL(MotorHAL):
    """
    Motor control via serial/UART protocol.

    Sends a simple text command each cycle:
      "L<pct>R<pct>\\n"   where pct is -100 to +100 integer percentage

    This is compatible with simple Arduino-based motor controllers.
    For Sabertooth, set MOTOR_SERIAL_BAUD=9600 and adapt the protocol below.
    """

    def __init__(self) -> None:
        try:
            import serial as _serial
        except ImportError:
            raise ImportError(
                "MOTOR_BACKEND=serial requires pyserial. "
                "Install with: pip3 install pyserial"
            )
        self._serial = _serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        self._left_speed  = 0.0
        self._right_speed = 0.0
        log.info("Serial motor HAL: port=%s baud=%d", SERIAL_PORT, SERIAL_BAUD)

    def apply(self, speed_left_ms: float, speed_right_ms: float) -> None:
        left_pct  = int(speed_left_ms  / MAX_SPEED_MS * 100)
        right_pct = int(speed_right_ms / MAX_SPEED_MS * 100)
        left_pct  = max(-100, min(100, left_pct))
        right_pct = max(-100, min(100, right_pct))
        cmd = f"L{left_pct:+04d}R{right_pct:+04d}\n".encode()
        try:
            self._serial.write(cmd)
        except Exception as exc:
            log.warning("Serial write failed: %s", exc)
        self._left_speed  = speed_left_ms
        self._right_speed = speed_right_ms

    def read(self) -> Tuple[float, float]:
        return self._left_speed, self._right_speed

    def stop(self) -> None:
        self.apply(0.0, 0.0)

    def close(self) -> None:
        self.stop()
        self._serial.close()
        log.info("Serial motor HAL closed")


def create_motor_hal() -> MotorHAL:
    backend = MOTOR_BACKEND.lower()
    if backend == "gpio":
        return GPIOMotorHAL()
    elif backend == "serial":
        return SerialMotorHAL()
    elif backend == "simulated":
        return None  # signal to MotorModel to use its built-in simulation
    else:
        raise ValueError(
            f"Unknown MOTOR_BACKEND={MOTOR_BACKEND!r}. "
            f"Valid options: simulated, gpio, serial"
        )


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
    """
    Per-rover motor state with first-order dynamics.

    When a hardware HAL is provided the apply_command() method routes to
    the physical HAL instead of the simulation model.  State estimation
    (temperature, RPM, current) always uses the simulation because physical
    sensors are not assumed to be present.
    """

    def __init__(self, rover_id: str, hal: Optional[MotorHAL] = None) -> None:
        self.rover_id      = rover_id
        self._hal          = hal
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
            if self._hal:
                self._hal.stop()
            return

        v = max(-MAX_SPEED_MS, min(MAX_SPEED_MS, cmd.target_speed_ms))
        w = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, cmd.turn_rate))

        # differential drive kinematics
        v_left  = v - w * WHEEL_BASE_M / 2
        v_right = v + w * WHEEL_BASE_M / 2

        if self._hal:
            # Hardware path: apply directly, read back actual speeds
            self._hal.apply(v_left, v_right)
            actual_left, actual_right = self._hal.read()
            self.speed_left  = actual_left
            self.speed_right = actual_right
        else:
            # Simulation path: first-order lag τ ≈ 0.1 s
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
_hal: Optional[MotorHAL] = None  # set once at startup; shared by all MotorModel instances


def _get_model(rover_id: str) -> MotorModel:
    if rover_id not in _models:
        _models[rover_id] = MotorModel(rover_id, hal=_hal)
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
    global _tracer, _hal
    start_http_server(METRICS_PORT)
    _tracer = setup_otel()

    # Initialise hardware HAL (may be None for simulated backend)
    if MOTOR_BACKEND != "simulated":
        log.info("Initialising motor hardware backend: %s", MOTOR_BACKEND)
        try:
            _hal = create_motor_hal()
        except Exception as exc:
            log.critical("Motor HAL init failed (%s): %s — aborting", MOTOR_BACKEND, exc)
            raise
    else:
        log.info("Motor backend: simulated (no physical hardware)")

    server = grpclib.server.Server([MotorControllerService()])
    await server.start("0.0.0.0", GRPC_PORT)
    log.info("MotorController gRPC server listening on :%d  backend=%s",
             GRPC_PORT, MOTOR_BACKEND)
    try:
        await server.wait_closed()
    finally:
        if _hal:
            _hal.close()


if __name__ == "__main__":
    asyncio.run(main())
