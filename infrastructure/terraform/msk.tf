# AWS MSK (Managed Streaming for Kafka) — handles 180K+ msg/sec
# with 3 brokers across 3 AZs for HA, m5.xlarge per broker.

resource "aws_msk_cluster" "rover_fleet" {
  cluster_name           = "${var.cluster_name}-kafka"
  kafka_version          = var.kafka_version
  number_of_broker_nodes = var.kafka_broker_count

  broker_node_group_info {
    instance_type   = var.kafka_instance_type
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.msk.id]

    storage_info {
      ebs_storage_info {
        volume_size = var.kafka_ebs_volume_size
        provisioned_throughput {
          enabled           = true
          volume_throughput = 250  # MiB/s per broker
        }
      }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.rover_fleet.arn
    revision = aws_msk_configuration.rover_fleet.latest_revision
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS_PLAINTEXT"
      in_cluster    = true
    }
  }

  logging {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
    }
  }

  open_monitoring {
    prometheus {
      jmx_exporter  { enabled_in_broker = true }
      node_exporter { enabled_in_broker = true }
    }
  }
}

resource "aws_msk_configuration" "rover_fleet" {
  name              = "${var.cluster_name}-kafka-config"
  kafka_versions    = [var.kafka_version]
  server_properties = <<-EOT
    # Throughput-optimised for 180K+ msg/sec
    num.network.threads=8
    num.io.threads=16
    socket.send.buffer.bytes=102400
    socket.receive.buffer.bytes=102400
    socket.request.max.bytes=104857600
    num.partitions=16
    num.recovery.threads.per.data.dir=2
    offsets.topic.replication.factor=3
    transaction.state.log.replication.factor=3
    transaction.state.log.min.isr=2
    log.retention.hours=24
    log.segment.bytes=1073741824
    log.retention.check.interval.ms=300000
    auto.create.topics.enable=false
    default.replication.factor=3
    min.insync.replicas=2
    # Low latency settings
    replica.fetch.max.bytes=10485760
    message.max.bytes=10485760
    compression.type=snappy
  EOT
}

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${var.cluster_name}"
  retention_in_days = 7
}
