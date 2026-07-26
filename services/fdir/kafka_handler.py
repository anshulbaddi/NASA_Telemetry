"""
Kafka consumer + producer for the FDIR service.

Consumes rover.telemetry (protobuf SensorData), runs anomaly detection,
and publishes FaultEvent + RecoveryCommand as protobuf bytes.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Callable, Optional

from confluent_kafka import Consumer, KafkaError, Producer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generated.telemetry.v1.telemetry_pb2 import SensorData
from generated.fdir.v1.fdir_pb2 import FaultEvent, RecoveryCommand, RecoveryAction

logger = logging.getLogger(__name__)


class KafkaFDIRHandler:
    IN_TOPIC          = "rover.telemetry"
    FDIR_EVENTS_TOPIC = "rover.fdir.events"
    COMMANDS_TOPIC    = "rover.commands"

    def __init__(self, brokers: str, group_id: str = "fdir-group"):
        self._consumer = Consumer({
            "bootstrap.servers":    brokers,
            "group.id":             group_id,
            "auto.offset.reset":    "latest",
            "enable.auto.commit":   True,
            "fetch.min.bytes":      65536,
            "fetch.wait.max.ms":    10,
            "max.poll.interval.ms": 300_000,
            "session.timeout.ms":   30_000,
        })
        self._producer = Producer({
            "bootstrap.servers": brokers,
            "compression.type":  "snappy",
            "linger.ms":         5,
            "batch.size":        65536,
            "acks":              1,
        })
        self._consumer.subscribe([self.IN_TOPIC])
        logger.info("FDIR Kafka handler ready topic=%s", self.IN_TOPIC)

    def run(
        self,
        on_message: Callable[[SensorData], Optional[FaultEvent]],
        stop_event,
    ) -> None:
        """Poll indefinitely; on_message returns a FaultEvent proto or None."""
        while not stop_event.is_set():
            msg = self._consumer.poll(timeout=0.05)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("kafka error: %s", msg.error())
                continue

            try:
                frame = SensorData().parse(msg.value())
                fault_event = on_message(frame)
                if fault_event:
                    self._publish_proto(self.FDIR_EVENTS_TOPIC, frame.rover_id, fault_event)
                    recovery = self._build_recovery(fault_event)
                    self._publish_proto(self.COMMANDS_TOPIC, frame.rover_id, recovery)
            except Exception as exc:
                logger.exception("message processing error: %s", exc)

        self._consumer.close()
        self._producer.flush(timeout=5)

    def _publish_proto(self, topic: str, key: str, value) -> None:
        self._producer.produce(
            topic,
            key           = key.encode(),
            value         = bytes(value),
            on_delivery   = self._delivery_report,
        )
        self._producer.poll(0)

    def _build_recovery(self, fault_event: FaultEvent) -> RecoveryCommand:
        return RecoveryCommand(
            rover_id     = fault_event.rover_id,
            action       = fault_event.recommended_action,
            reason       = fault_event.description,
            issued_at_ms = int(time.time() * 1000),
            acknowledged = False,
        )

    @staticmethod
    def _delivery_report(err, msg) -> None:
        if err:
            logger.error("delivery failed: %s [%s]", err, msg.topic())
