// main.c - driver generico de inferencia p/ gem5 (ROI via m5_dump_stats)
// Compilar definindo MODEL_FILE, ex.:
//   gcc -O2 -static -DMODEL_FILE='"rf_model.h"' main.c ... -o rf_bin
#include <stdio.h>
#include <stdlib.h>
#include "dataset.h"   // define NUM_FEATURES (usado pelo predict do MODEL_FILE)
#include MODEL_FILE     // define predict() e NUM_CLASSES
#include <gem5/m5ops.h>

int main(int argc, char *argv[]) {
    int slice_id = (argc > 1) ? atoi(argv[1]) : 0;
    int num_cpus = (argc > 2) ? atoi(argv[2]) : 1;

    int per_core = NUM_SAMPLES / num_cpus;
    int start = slice_id * per_core;
    int end = (slice_id == num_cpus - 1) ? NUM_SAMPLES : start + per_core;

    int dummy = 0;

    // warmup (fora da ROI)
    int warm = 100; if (warm > NUM_SAMPLES) warm = NUM_SAMPLES;
    for (int i = 0; i < warm; i++) dummy += predict(can_data[i]);

    if (slice_id == 0) m5_dump_reset_stats(0, 0);

    // REGIAO DE INTERESSE
    for (int i = start; i < end; i++) dummy += predict(can_data[i]);

    if (slice_id == num_cpus - 1) m5_dump_stats(0, 0);

    printf("[Core %d] %d..%d dummy=%d\n", slice_id, start, end - 1, dummy);
    return 0;
}
