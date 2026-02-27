#!/usr/bin/env sh
set -eu

BRANCH="${DEPLOY_BRANCH:-main}"
API_SERVICE="${API_SERVICE:-donkeyseo-api}"
WORKER_SERVICE="${WORKER_SERVICE:-donkeyseo-worker}"
TUNNEL_SERVICE="${TUNNEL_SERVICE:-cloudflared-donkeyseo}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

unit_exists() {
  unit="$1"
  unit_type="$2"
  systemctl list-unit-files --type="$unit_type" --no-legend 2>/dev/null | awk '{print $1}' | grep -qx "$unit"
}

require_cmd() {
  cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    exit 1
  fi
}

require_cmd git
require_cmd docker
require_cmd uv
require_cmd curl
require_cmd systemctl

compose_up_service_allow_port_in_use() {
  service="$1"
  port="$2"
  service_label="$3"

  set +e
  up_output="$(docker compose up -d "${service}" 2>&1)"
  up_status=$?
  set -e

  if [ "${up_status}" -eq 0 ]; then
    printf '%s\n' "${up_output}"
  elif printf '%s\n' "${up_output}" | grep -q ":${port}: bind: address already in use"; then
    echo "${service_label} port ${port} is already in use; skipping ${service} container startup."
    echo "Continuing deploy with the existing ${service_label} instance."
  else
    printf '%s\n' "${up_output}"
    exit "${up_status}"
  fi
}

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree has uncommitted changes. Commit/stash them before deploying."
  exit 1
fi

echo "Fetching latest code (${BRANCH})..."
git fetch origin "${BRANCH}"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "${CURRENT_BRANCH}" != "${BRANCH}" ]; then
  if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git switch "${BRANCH}"
  else
    git switch -c "${BRANCH}" --track "origin/${BRANCH}"
  fi
fi

git pull --ff-only origin "${BRANCH}"

echo "Ensuring Postgres/Redis are running..."
compose_up_service_allow_port_in_use db 5432 Postgres
compose_up_service_allow_port_in_use redis 6379 Redis

echo "Syncing Python dependencies..."
uv sync --frozen

echo "Running database migrations..."
uv run alembic upgrade head

echo "Reloading systemd units..."
sudo systemctl daemon-reload

echo "Restarting services..."
sudo systemctl restart "${API_SERVICE}" "${WORKER_SERVICE}"

if unit_exists "${TUNNEL_SERVICE}" service; then
  sudo systemctl restart "${TUNNEL_SERVICE}"
else
  echo "Tunnel service not found, skipping restart (${TUNNEL_SERVICE})"
fi

echo "Service states:"
sudo systemctl is-active "${API_SERVICE}"
sudo systemctl is-active "${WORKER_SERVICE}"
if unit_exists "${TUNNEL_SERVICE}" service; then
  sudo systemctl is-active "${TUNNEL_SERVICE}"
fi

echo "Local health checks:"
curl -fsS http://127.0.0.1:8000/health && echo
curl -fsS http://127.0.0.1:8000/health/queue && echo

if [ -n "${PUBLIC_HOST:-}" ]; then
  echo "Public health check (${PUBLIC_HOST}):"
  curl -fsS "https://${PUBLIC_HOST}/health" && echo
fi

if [ "${FOLLOW_LOGS}" = "1" ]; then
  echo "Following logs (Ctrl+C to stop)..."
  if unit_exists "${TUNNEL_SERVICE}" service; then
    sudo journalctl -u "${API_SERVICE}" -u "${WORKER_SERVICE}" -u "${TUNNEL_SERVICE}" -f
  else
    sudo journalctl -u "${API_SERVICE}" -u "${WORKER_SERVICE}" -f
  fi
fi

echo "Deploy finished successfully."
