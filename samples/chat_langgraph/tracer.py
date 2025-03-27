
import os
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace, trace as trace_api
from opentelemetry.trace import Tracer
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

class AppInsightsTracer():

    def __init__(self):
        self.tracer = self._setup_tracing()

    def get_tracer(self) -> Tracer:
        return self.tracer

    def _setup_tracing(self):
        exporter = AzureMonitorTraceExporter.from_connection_string(
            os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
        )
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(__name__)
        span_processor = BatchSpanProcessor(exporter, schedule_delay_millis=60000)
        trace.get_tracer_provider().add_span_processor(span_processor)
        LangchainInstrumentor().instrument()
        return tracer