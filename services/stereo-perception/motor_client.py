"""
Non-blocking motor command client for the stereo-perception control loop.

Sends drive commands to the motor-controller gRPC service running on the same
machine (Jetson Nano) without blocking the perception/safety thread.

Architecture:
  - One background asyncio event loop runs in a daemon thread.
  - The control loop calls send() which schedules the coroutine onto that loop
    using run_coroutine_threadsafe() and returns immediately.
  - If the motor-controller is unreachable, failures are logged but never
    block or crash the perception pipeline.

On Jetson hardware the motor-controller gRPC service runs with
MOTOR_BACKEND=gpio or MOTOR_BACKEND=serial (see services/motor-controller/main.py).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../.."))

try:
    import grpclib.client
    from generated.motorcontroller.v1.motor_controller_pb2 import (
        MotorControllerStub, DriveCommand,
    )
    _GRPCLIB = True
except ImportError:
    _GRPCLIB = False
    log.warning("grpclib not available; motor commands will be simulated locally")


# ── Direction → (speed_ms, turn_rate) mapping ─────────────────────────────────

_DIRECTION_MAP = {
    "FORWARD":       (1.0,  0.0),   # factors applied to config.MAX_SPEED_MS
    "TURN_LEFT":     (0.3, -0.6),
    "TURN_RIGHT":    (0.3,  0.6),
    "STOP":          (0.0,  0.0),
    "EMERGENCY_STOP":(0.0,  0.0),
    "REVERSE":       (-0.5, 0.0),
}


class MotorCommandClient:
    """
    Sends DriveCommand protobuf messages to the motor-controller gRPC service.

    Usage:
        client = MotorCommandClient(config)
        client.start()
        client.send("FORWARD")          # non-blocking
        client.send("EMERGENCY_STOP")   # non-blocking, sets emergency_stop=True
        client.stop()
    """

    def __init__(self, config) -> None:
        self._addr        = config.MOTOR_CONTROLLER_ADDR
        self._rover_id    = config.ROVER_ID
        self._max_speed   = config.MAX_SPEED_MS
        self._turn_rate   = config.TURN_RATE
        self._timeout     = config.MOTOR_COMMAND_TIMEOUT_S
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._last_command = "STOP"
        self._last_send_ts = 0.0

    def start(self) -> None:
        if not _GRPCLIB:
            log.warning("Motor client: grpclib unavailable, running in no-op mode")
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="motor-grpc-loop",
        )
        self._thread.start()
        log.info("MotorCommandClient started: addr=%s rover=%s", self._addr, self._rover_id)

    def send(self, direction: str) -> None:
        """
        Send a drive command.  Non-blocking — returns immediately.

        Repeated identical commands are deduplicated unless they are
        EMERGENCY_STOP to avoid spamming the gRPC service.
        """
        if direction != "EMERGENCY_STOP" and direction == self._last_command:
            # Only resend periodically even if command is the same
            if time.time() - self._last_send_ts < 0.5:
                return

        self._last_command = direction
        self._last_send_ts = time.time()

        if self._loop is None:
            return

        asyncio.run_coroutine_threadsafe(
            self._send_async(direction),
            self._loop,
        )

    async def _send_async(self, direction: str) -> None:
        speed_factor, turn_factor = _DIRECTION_MAP.get(direction, (0.0, 0.0))
        speed_ms  = speed_factor * self._max_speed
        turn_rate = turn_factor  * self._turn_rate
        is_estop  = (direction == "EMERGENCY_STOP")

        host, port_str = self._addr.rsplit(":", 1)
        try:
            async with grpclib.client.Channel(host, int(port_str)) as channel:
                stub = MotorControllerStub(channel)
                cmd  = DriveCommand(
                    rover_id        = self._rover_id,
                    target_speed_ms = speed_ms,
                    turn_rate       = turn_rate,
                    duration_ms     = 200.0,   # 200 ms per command tick
                    emergency_stop  = is_estop,
                )
                if is_estop:
                    await stub.emergency_stop(cmd)
                else:
                    await stub.send_command(cmd)
        except Exception as exc:
            log.warning("Motor command failed (direction=%s): %s", direction, exc)

    def stop(self) -> None:
        """Send a final STOP command and shut down the background loop."""
        if self._loop and self._loop.is_running():
            # Fire final stop
            asyncio.run_coroutine_threadsafe(
                self._send_async("EMERGENCY_STOP"), self._loop
            ).result(timeout=2.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3.0)
        log.info("MotorCommandClient stopped")
