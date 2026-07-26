"""OpenTelemetry initialisation for the FDIR service."""
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION


def setup(service_name: str = "fdir") -> trace.Tracer:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    version = os.getenv("OTEL_SERVICE_VERSION", "dev")

    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: version,
    })

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter, max_export_batch_size=256))
    trace.set_tracer_provider(provider)

    return trace.get_tracer(service_name)
