#!/usr/bin/env bash
# Jetson Nano rover install script.
# Run once after cloning the repo onto the Jetson:
#   cd /home/jetson/rover
#   bash deployment/jetson/install.sh
set -euo pipefail

ROVER_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROVER_DIR/venv"
SYSTEMD_DIR="/etc/systemd/system"
DEPLOY="$ROVER_DIR/deployment/jetson"

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

[[ "$(uname -m)" == "aarch64" ]] || warn "Not running on aarch64 — some packages may behave differently"
[[ $EUID -eq 0 ]]                 || die "Run as root: sudo bash deployment/jetson/install.sh"

# ── System packages ─────────────────────────────────────────────────────────
info "Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv \
    libyaml-dev \
    libssl-dev libffi-dev \
    gcc g++ make \
    git curl \
    v4l-utils

# ── Python venv ─────────────────────────────────────────────────────────────
info "Creating Python venv at $VENV..."
python3 -m venv "$VENV" --system-site-packages
"$VENV/bin/pip" install --upgrade pip wheel

# ── Python dependencies ──────────────────────────────────────────────────────
info "Installing stereo-perception requirements..."
"$VENV/bin/pip" install -r "$ROVER_DIR/services/stereo-perception/requirements.txt"

info "Installing motor-controller requirements..."
"$VENV/bin/pip" install -r "$ROVER_DIR/services/motor-controller/requirements.txt"

# ── Env file ─────────────────────────────────────────────────────────────────
if [[ ! -f "$ROVER_DIR/.env.rover" ]]; then
    info "Copying .env.rover.example → .env.rover"
    cp "$DEPLOY/.env.rover.example" "$ROVER_DIR/.env.rover"
    warn "Edit $ROVER_DIR/.env.rover and set GROUND_STATION_IP before starting services"
else
    info ".env.rover already exists — skipping copy"
fi

# ── Config directory ──────────────────────────────────────────────────────────
mkdir -p "$ROVER_DIR/config"
if [[ ! -f "$ROVER_DIR/config/stereo_calibration.yaml" ]]; then
    info "Copying placeholder calibration to config/stereo_calibration.yaml"
    cp "$ROVER_DIR/config/stereo_calibration.yaml" "$ROVER_DIR/config/stereo_calibration.yaml" 2>/dev/null || true
    warn "Replace config/stereo_calibration.yaml with real calibration data (see docs/calibration_guide.md)"
fi

# ── Data directories ─────────────────────────────────────────────────────────
mkdir -p "$ROVER_DIR/data/test_stereo/left"
mkdir -p "$ROVER_DIR/data/test_stereo/right"

# ── systemd units ────────────────────────────────────────────────────────────
info "Installing systemd units..."
install -m 644 "$DEPLOY/motor-controller.service"    "$SYSTEMD_DIR/motor-controller.service"
install -m 644 "$DEPLOY/stereo-perception.service"   "$SYSTEMD_DIR/stereo-perception.service"

systemctl daemon-reload
systemctl enable motor-controller.service stereo-perception.service
info "Services enabled (not started — run 'sudo systemctl start motor-controller stereo-perception')"

# ── gpio group ───────────────────────────────────────────────────────────────
if ! getent group gpio > /dev/null 2>&1; then
    info "Creating gpio group..."
    groupadd gpio
fi
usermod -aG gpio jetson 2>/dev/null || warn "User 'jetson' not found — add rover user to gpio group manually"

# ── Permissions on /dev/gpiochip* ────────────────────────────────────────────
UDEV_RULE="/etc/udev/rules.d/99-gpio.rules"
if [[ ! -f "$UDEV_RULE" ]]; then
    info "Writing udev rule for GPIO access..."
    cat > "$UDEV_RULE" <<'EOF'
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
SUBSYSTEM=="gpio", KERNEL=="gpio*",     GROUP="gpio", MODE="0660"
EOF
    udevadm control --reload-rules
    udevadm trigger
fi

info "────────────────────────────────────────────────────────────────────"
info "Installation complete."
info ""
info "Next steps:"
info "  1. Edit $ROVER_DIR/.env.rover"
info "     — Set GROUND_STATION_IP to your ground station's LAN IP"
info "     — Set CAMERA_TYPE (csi/usb/file)"
info "     — Set MOTOR_BACKEND (gpio/serial/simulated)"
info "  2. Run the camera calibration (if using a real camera):"
info "     python3 scripts/calibrate_stereo.py auto --output config/stereo_calibration.yaml"
info "     (See docs/calibration_guide.md for step-by-step instructions)"
info "  3. Start services:"
info "     sudo systemctl start motor-controller stereo-perception"
info "  4. Check logs:"
info "     journalctl -fu stereo-perception"
info "     journalctl -fu motor-controller"
info "────────────────────────────────────────────────────────────────────"
