#!/bin/bash

# AIcheckers Bot Control Script
# Usage: sh scripts/bot_control.sh [on|off|status]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLAG_FILE="$SCRIPT_DIR/.bot_active"

case "$1" in
    on)
        touch "$FLAG_FILE"
        echo "✅ BotをONにしました。監視を再開します。"
        ;;
    off)
        rm -f "$FLAG_FILE"
        echo "🛑 BotをOFFにしました。監視を停止します。"
        ;;
    status)
        if [ -f "$FLAG_FILE" ]; then
            echo "🟢 現在の状態: ON (監視中)"
        else
            echo "🔴 現在の状態: OFF (停止中)"
        fi
        ;;
    *)
        echo "Usage: $0 {on|off|status}"
        exit 1
        ;;
esac
