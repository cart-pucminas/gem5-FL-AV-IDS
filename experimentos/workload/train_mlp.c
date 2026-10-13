// train_mlp.c - micro-benchmark do TREINO LOCAL de 1 round do MLP (Federated
// Learning on-device). Executa as operacoes DOMINANTES de um round de treino:
//   forward (matmul + ReLU) -> loss -> backprop (matmul transposta) -> update SGD
// sobre um minibatch de BATCH_SIZE amostras, por EPOCHS_ROI iteracoes (a ROI).
//
// Objetivo: caracterizar o PERFIL ARQUITETURAL do treino (FP-intensivo, matmul,
// padrao de memoria) - NAO produzir um modelo convergido. Usa os pesos reais do
// mlp_model.h como ponto de partida e o dataset.h como dados de entrada.
//
// Compilar (define BATCH_SIZE: 32/64/128) com main de treino, ex.:
//   aarch64-linux-gnu-gcc -O2 -static -DBATCH_SIZE=64 \
//     -DMODEL_FILE='"mlp_model.h"' -I<gem5>/include -Ic_export \
//     train_mlp.c <gem5>/util/m5/build/arm64/out/libm5.a -o bin/train_mlp_b64
//
// A ROI (m5_dump) cobre os rounds de treino; o forward de warmup fica fora.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>         // expf (softmax) - mantem saidas em [0,1], evita NaN
#include "dataset.h"      // NUM_FEATURES, NUM_SAMPLES, can_data[][]
#include MODEL_FILE        // NUM_CLASSES, N_LAYERS, layer_dims[], sc_mean/scale, W0..,b0..
#include <gem5/m5ops.h>

#ifndef BATCH_SIZE
#define BATCH_SIZE 64
#endif
#ifndef EPOCHS_ROI
#define EPOCHS_ROI 50      // nro de rounds/iteracoes de treino medidos na ROI
#endif
#define LR 0.0001f    // baixo p/ manter os gradientes estaveis (evita explosao/NaN)

// Copias TREINAVEIS dos pesos (nao alteram o mlp_model.h original).
// Dimensoes: layer_dims = {9,64,32,5}. Alocamos com MAX_DIM p/ simplicidade.
static float w[N_LAYERS][MAX_DIM * MAX_DIM];
static float b[N_LAYERS][MAX_DIM];
static float *W_src[N_LAYERS];   // pesos iniciais (do header)
static float *b_src[N_LAYERS];

// buffers de ativacao/gradiente por camada
static float act[N_LAYERS + 1][MAX_DIM];   // act[0]=entrada; act[l+1]=saida camada l
static float delta[N_LAYERS][MAX_DIM];      // erro por camada

static void init_pesos(void) {
    float *Ws[N_LAYERS] = { W0, W1, W2 };
    float *bs[N_LAYERS] = { b0, b1, b2 };
    for (int l = 0; l < N_LAYERS; l++) {
        W_src[l] = Ws[l]; b_src[l] = bs[l];
        int din = layer_dims[l], dout = layer_dims[l + 1];
        for (int i = 0; i < din * dout; i++) w[l][i] = Ws[l][i];
        for (int j = 0; j < dout; j++) b[l][j] = bs[l][j];
    }
}

// forward: preenche act[]; retorna via act[N_LAYERS]
static void forward(const float *x) {
    for (int i = 0; i < NUM_FEATURES; i++)
        act[0][i] = (x[i] - sc_mean[i]) / sc_scale[i];
    for (int l = 0; l < N_LAYERS; l++) {
        int din = layer_dims[l], dout = layer_dims[l + 1];
        for (int j = 0; j < dout; j++) {
            float s = b[l][j];
            for (int i = 0; i < din; i++) s += act[l][i] * w[l][i * dout + j];
            act[l + 1][j] = (l < N_LAYERS - 1 && s < 0) ? 0 : s;  // ReLU exceto saida
        }
    }
    // softmax na saida (classificacao) - mantem valores em [0,1], evita NaN
    // e torna o perfil de compute mais fiel a um treino real (inclui expf).
    int Lo = N_LAYERS, dout = layer_dims[N_LAYERS];
    float mx = act[Lo][0];
    for (int j = 1; j < dout; j++) if (act[Lo][j] > mx) mx = act[Lo][j];
    float sum = 0;
    for (int j = 0; j < dout; j++) { act[Lo][j] = expf(act[Lo][j] - mx); sum += act[Lo][j]; }
    for (int j = 0; j < dout; j++) act[Lo][j] /= (sum + 1e-9f);
}

// backward + update SGD para uma amostra (rotulo y). Perfil dominante do treino.
static void backward_update(int y) {
    int L = N_LAYERS - 1;
    int dout = layer_dims[L + 1];
    // delta da saida (softmax/MSE-like simplificado: (out - onehot))
    for (int j = 0; j < dout; j++)
        delta[L][j] = act[L + 1][j] - (j == y ? 1.0f : 0.0f);
    // propaga para tras
    for (int l = L; l >= 0; l--) {
        int din = layer_dims[l], do_ = layer_dims[l + 1];
        // gradiente para camada anterior (matmul transposta)
        if (l > 0) {
            for (int i = 0; i < din; i++) {
                float s = 0;
                for (int j = 0; j < do_; j++) s += delta[l][j] * w[l][i * do_ + j];
                delta[l - 1][i] = (act[l][i] > 0) ? s : 0;   // deriv ReLU
            }
        }
        // update dos pesos (SGD)
        for (int i = 0; i < din; i++)
            for (int j = 0; j < do_; j++)
                w[l][i * do_ + j] -= LR * delta[l][j] * act[l][i];
        for (int j = 0; j < do_; j++)
            b[l][j] -= LR * delta[l][j];
    }
}

int main(int argc, char *argv[]) {
    int slice_id = (argc > 1) ? atoi(argv[1]) : 0;
    int num_cpus = (argc > 2) ? atoi(argv[2]) : 1;
    (void)slice_id; (void)num_cpus;   // treino local roda 1 core (FL client)

    init_pesos();

    int batch = BATCH_SIZE;
    if (batch > NUM_SAMPLES) batch = NUM_SAMPLES;

    // rotulos sinteticos deterministicos (nao temos y no dataset.h; para o perfil
    // de compute do backprop o valor do rotulo e irrelevante - so precisa existir)
    // usa (indice % NUM_CLASSES).

    // warmup: 1 round fora da ROI
    for (int i = 0; i < batch; i++) { forward(can_data[i]); backward_update(i % NUM_CLASSES); }

    m5_dump_reset_stats(0, 0);
    // ===== ROI: EPOCHS_ROI rounds de treino local sobre o minibatch =====
    volatile float sink = 0;
    for (int e = 0; e < EPOCHS_ROI; e++) {
        for (int i = 0; i < batch; i++) {
            forward(can_data[i]);
            backward_update(i % NUM_CLASSES);
            sink += act[N_LAYERS][0];
        }
    }
    m5_dump_stats(0, 0);

    printf("[Treino MLP] batch=%d epochs=%d sink=%f\n", batch, EPOCHS_ROI, (float)sink);
    return 0;
}
