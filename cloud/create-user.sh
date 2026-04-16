#!/usr/bin/env bash
set -euo pipefail

NEON_AUTH_URL="https://ep-shiny-shape-aj0git9l.neonauth.c-3.us-east-2.aws.neon.tech/neondb/auth"

read -rp "Email: " email
read -rp "Name: " name
read -rsp "Password: " password
echo

response=$(curl -s -w "\n%{http_code}" \
  -X POST "${NEON_AUTH_URL}/sign-up/email" \
  -H "Content-Type: application/json" \
  -H "Origin: http://localhost:5173" \
  -d "{\"email\":\"${email}\",\"password\":\"${password}\",\"name\":\"${name}\"}")

http_code=$(echo "$response" | tail -1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
  echo "User created: ${email}"
else
  echo "Failed (HTTP ${http_code}): ${body}"
  exit 1
fi
