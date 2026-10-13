#!/bin/bash
# Compila o micro-benchmark de TREINO local do MLP (train_mlp.c) em 3 tamanhos de
# batch (32/64/128), para ARM aarch64 estatico. Usa os pesos reais do mlp_model.h.
#
# Rodar de dentro de experimentos/workload/.
set -e

GEM5="${GEM5:-../../gem5}"
CC="${CC:-aarch64-linux-gnu-gcc}"
INCLUDE="$GEM5/include"
LIBM5="$GEM5/util/m5/build/arm64/out/libm5.a"
SRC="train_mlp.c"
HDRS="c_export"          # onde estao mlp_model.h e dataset.h
OUT="c_export/bin"

mkdir -p "$OUT"
command -v "$CC" >/dev/null 2>&1 || { echo "[ERRO] $CC nao encontrado"; exit 1; }
[ -f "$LIBM5" ] || { echo "[ERRO] libm5 nao encontrada: $LIBM5"; exit 1; }

for B in 32 64 128; do
    echo "[*] compilando treino MLP batch=$B ..."
    "$CC" -O2 -static \
        -DBATCH_SIZE=$B \
        -DMODEL_FILE='"mlp_model.h"' \
        -I"$INCLUDE" -I"$HDRS" \
        "$SRC" "$LIBM5" -lm \
        -o "$OUT/train_mlp_b${B}"
    echo "    -> $OUT/train_mlp_b${B}"
done

echo ""
ls -la "$OUT"/train_mlp_b* 2>/dev/null
file "$OUT"/train_mlp_b* 2>/dev/null | head
