#!/bin/bash
# Compila os 4 modelos (RF, XGBoost, MLP, SVM) para binarios ARM aarch64 estaticos,
# prontos p/ rodar no gem5. Cada binario usa main.c + <modelo>_model.h.
#
# Requer: aarch64-linux-gnu-gcc (mesmo cross-compiler usado no ERAD; os binarios
# antigos sao "ELF 64-bit ARM aarch64, statically linked").
#
# Rodar de dentro de experimentos/workload/ (onde estao main.c e c_export/).
# Ajuste GEM5 se o caminho do gem5 for outro.
set -e

GEM5="${GEM5:-../../gem5}"                 # raiz do gem5 (relativa a workload/)
CC="${CC:-aarch64-linux-gnu-gcc}"
INCLUDE="$GEM5/include"
LIBM5="$GEM5/util/m5/build/arm64/out/libm5.a"
SRC="c_export/main.c"
HDRS="c_export"                            # onde estao os *_model.h e dataset.h
OUT="c_export/bin"

mkdir -p "$OUT"

if ! command -v "$CC" >/dev/null 2>&1; then
    echo "[ERRO] '$CC' nao encontrado. Instale o cross-compiler aarch64 ou ajuste CC=."
    echo "  Debian/Ubuntu: sudo apt install gcc-aarch64-linux-gnu"
    exit 1
fi
[ -f "$LIBM5" ] || { echo "[ERRO] libm5 nao encontrada: $LIBM5"; exit 1; }

for m in rf xgb mlp svm; do
    hdr="$HDRS/${m}_model.h"
    if [ ! -f "$hdr" ]; then
        echo "[skip] $hdr nao existe"; continue
    fi
    echo "[*] compilando $m ..."
    "$CC" -O2 -static \
        -DMODEL_FILE="\"${m}_model.h\"" \
        -I"$INCLUDE" -I"$HDRS" \
        "$SRC" "$LIBM5" \
        -o "$OUT/${m}_arm"
    echo "    -> $OUT/${m}_arm"
done

echo ""
echo "Binarios em $OUT/ :"
ls -la "$OUT"
echo ""
echo "Confira a arquitetura (deve ser 'ARM aarch64, statically linked'):"
file "$OUT"/*_arm 2>/dev/null | head
