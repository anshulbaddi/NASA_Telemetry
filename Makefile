SHELL    := /bin/bash
REGISTRY ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
TAG      ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")
SERVICES := producer consumer path-planner obstacle-detector motor-controller api-gateway fdir
PYTHON   ?= python3

.PHONY: all dev dev-down install proto proto-lint docker-build docker-push ecr-login \
        k8s-apply k8s-rollout-status k8s-delete \
        infra-init infra-plan infra-apply infra-destroy \
        test lint clean help

all: docker-build

## ── Local dev ────────────────────────────────────────────────────────────────
dev:
	docker compose up --build

dev-down:
	docker compose down -v

## ── Python deps (host install for local testing) ─────────────────────────────
install:
	@for svc in $(SERVICES); do \
		echo "==> pip install services/$$svc"; \
		pip install -q -r services/$$svc/requirements.txt; \
	done

## ── Proto generation (betterproto) ──────────────────────────────────────────
proto:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 required"; exit 1; }
	pip install -q grpcio-tools betterproto[compiler]==0.3.0
	@for proto in $$(find proto -name '*.proto'); do \
		pkg=$$(dirname $$proto | sed 's|proto/||'); \
		out=generated/$$(echo $$pkg | tr '/' '/'); \
		mkdir -p $$out; \
		python3 -m grpc_tools.protoc \
			-I proto \
			--python_betterproto_out=generated \
			$$proto; \
		echo "Generated: $$proto → $$out"; \
	done

proto-lint:
	@command -v buf >/dev/null 2>&1 && buf lint || echo "buf not installed; skipping lint"

## ── Native extensions (Cython / C++) ─────────────────────────────────────────
build-ext:
	@echo "==> Building lidar_cluster (Cython)"
	cd services/obstacle-detector && python setup.py build_ext --inplace -q
	@echo "==> Building fdir_native (Cython + C++)"
	cd services/fdir && python setup.py build_ext --inplace -q

## ── Syntax / unit tests ──────────────────────────────────────────────────────
test:
	@echo "==> Python syntax check"
	@find services generated -name '*.py' | xargs $(PYTHON) -m py_compile && echo "  All files OK"
	@echo "==> YAML validation"
	@find k8s monitoring -name '*.yaml' -o -name '*.yml' | \
		xargs $(PYTHON) -c "import sys,yaml; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" && \
		echo "  All YAML OK"

## ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	@for svc in $(SERVICES); do \
		echo "==> Building image rover-$$svc:$(TAG)"; \
		docker build -f services/$$svc/Dockerfile -t rover-$$svc:$(TAG) -t rover-$$svc:latest .; \
	done

docker-push: docker-build
	@for svc in $(SERVICES); do \
		docker tag rover-$$svc:$(TAG) $(REGISTRY)/rover-$$svc:$(TAG); \
		docker push $(REGISTRY)/rover-$$svc:$(TAG); \
	done

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(REGISTRY)

## ── Kubernetes ───────────────────────────────────────────────────────────────
k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	@for svc in producer consumer path-planner obstacle-detector motor-controller fdir api-gateway; do \
		kubectl apply -f k8s/$$svc/; \
	done
	kubectl apply -f k8s/monitoring/

k8s-rollout-status:
	@for svc in producer consumer path-planner obstacle-detector motor-controller fdir api-gateway; do \
		kubectl rollout status deployment/$$svc -n rover-fleet; \
	done

k8s-delete:
	kubectl delete namespace rover-fleet --ignore-not-found

## ── Terraform ────────────────────────────────────────────────────────────────
infra-init:
	cd infrastructure/terraform && terraform init

infra-plan:
	cd infrastructure/terraform && terraform plan -out=tfplan

infra-apply:
	cd infrastructure/terraform && terraform apply tfplan

infra-destroy:
	cd infrastructure/terraform && terraform destroy

## ── Observability shortcuts ──────────────────────────────────────────────────
logs:
	kubectl logs -n rover-fleet -l app=$(SVC) -f --tail=100

port-forward-grafana:
	kubectl port-forward svc/grafana 3000:3000 -n rover-fleet

port-forward-jaeger:
	kubectl port-forward svc/jaeger 16686:16686 -n rover-fleet

port-forward-api:
	kubectl port-forward svc/api-gateway 8080:8080 -n rover-fleet

## ── Misc ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true

help:
	@grep -E '^[a-zA-Z_-]+:' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":"}; {printf "\033[36m%-22s\033[0m\n", $$1}'
