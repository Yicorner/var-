#!/bin/bash
set -e  # 遇错退出

# 目标持久目录
PERSIST_DIR="/home/featurize/work/.cursor-server"

# 如果原始目录存在且不是符号链接，则先迁移内容
mkdir -p "$PERSIST_DIR"
if [ -d "$HOME/.cursor-server" ] && [ ! -L "$HOME/.cursor-server" ]; then
  echo "Backing up existing ~/.cursor-server to $PERSIST_DIR ..."
  rsync -a "$HOME/.cursor-server/" "$PERSIST_DIR/"
  rm -rf "$HOME/.cursor-server"
fi

# 建立符号链接
ln -sfn "$PERSIST_DIR" "$HOME/.cursor-server"
echo "Linked ~/.cursor-server -> $PERSIST_DIR"

echo "✅ Done. Cursor server extensions will now persist under $PERSIST_DIR"
