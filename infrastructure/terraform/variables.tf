variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (production | staging)"
  type        = string
  default     = "production"
  validation {
    condition     = contains(["production", "staging"], var.environment)
    error_message = "environment must be 'production' or 'staging'."
  }
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "rover-fleet-prod"
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.30"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS node groups"
  type        = list(string)
  default     = ["m6i.xlarge", "m6a.xlarge"]
}

variable "node_min_size" {
  description = "Minimum number of EKS worker nodes"
  type        = number
  default     = 3
}

variable "node_max_size" {
  description = "Maximum number of EKS worker nodes (supports 5× traffic spike)"
  type        = number
  default     = 15
}

variable "node_desired_size" {
  description = "Desired number of EKS worker nodes"
  type        = number
  default     = 5
}

variable "kafka_instance_type" {
  description = "MSK broker instance type"
  type        = string
  default     = "kafka.m5.xlarge"
}

variable "kafka_broker_count" {
  description = "Number of MSK brokers (one per AZ)"
  type        = number
  default     = 3
}

variable "kafka_ebs_volume_size" {
  description = "EBS volume size per MSK broker in GiB"
  type        = number
  default     = 1000
}

variable "kafka_version" {
  description = "Apache Kafka version for MSK"
  type        = string
  default     = "3.6.0"
}

variable "ecr_repos" {
  description = "ECR repository names"
  type        = list(string)
  default = [
    "rover-producer",
    "rover-consumer",
    "rover-path-planner",
    "rover-obstacle-detector",
    "rover-motor-controller",
    "rover-fdir",
    "rover-api-gateway",
  ]
}
