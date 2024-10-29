#include <vector>
#include <cuda_runtime.h>
#include <string>
#include "headers/cuda_bpe.h"
#include "headers/bpe_seq.h"
#include <unordered_map>

cuda_data_t init_and_copy_data_to_cuda(const std::string &text) {
    int length = text.length();

    char *d_text;
    int *d_tokens;
    int *freq_mat, *tok2bp1, *tok2bp2;

    cudaMalloc((void **) &freq_mat, FREQ_MAT_SIZE * sizeof(int));
    cudaMalloc((void **) &tok2bp1, MAX_TOKENS * sizeof(int));
    cudaMalloc((void **) &tok2bp2, MAX_TOKENS * sizeof(int));

    cudaMemset(&freq_mat, 0, FREQ_MAT_SIZE * sizeof(int));
    cudaMemset(&tok2bp1, NA_TOKEN, MAX_TOKENS * sizeof(int));
    cudaMemset(&tok2bp2, NA_TOKEN, MAX_TOKENS * sizeof(int));

    cudaMalloc((void **) &d_text, length * sizeof(char));
    cudaMalloc((void **) &d_tokens, length * sizeof(int));

    //todo - check for memory alloc failure

    // Copy data from host to device
    cudaMemcpy(d_text, text.c_str(), length * sizeof(char), cudaMemcpyHostToDevice);


    cu_mem_t mem_dt = {d_text, d_tokens, length, length};
    cu_bpe_t bpe_t = {tok2bp1, tok2bp2, freq_mat};

    cu_bpe_processing_t *proc_dt;
    cudaMalloc((void **) &proc_dt, sizeof(cu_bpe_processing_t));

    return {mem_dt, bpe_t, proc_dt};
}

std::unordered_map<int, std::pair<int, int>> convert_data_to_map(int *tok2bp1, int *tok2bp2) {
    std::unordered_map<int, std::pair<int, int>> tok2bp;
    for (int i = 0; i < MAX_TOKENS; i++) {
        if ((tok2bp1[i] > 0) && (tok2bp1[i] != NA_TOKEN) && (tok2bp2[i] > 0) && (tok2bp2[i] != NA_TOKEN)) {
            tok2bp[i] = std::make_pair(tok2bp1[i], tok2bp2[i]);
        }
    }
    return tok2bp;
}

CommonData bring_data_from_cuda(cuda_data_t cdt) {
    int *h_tokens = new int[cdt.mem_dt.inp_text_length];
    int *tok2bp1 = new int[MAX_TOKENS];
    int *tok2bp2 = new int[MAX_TOKENS];

    cudaMemcpy(h_tokens, cdt.mem_dt.d_tokens, cdt.mem_dt.inp_text_length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tok2bp1, cdt.bpe_dt.tok2bp1, MAX_TOKENS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tok2bp2, cdt.bpe_dt.tok2bp2, MAX_TOKENS * sizeof(int), cudaMemcpyDeviceToHost);


    CommonData cd = CommonData(convert_data_to_map(tok2bp1, tok2bp2));
    cd.training_iters=cdt.train_iters;
    // Copy results to vector (since CUDA doesn't support vectors directly)
    std::vector<int> tokens(h_tokens, h_tokens + cdt.mem_dt.inp_text_length);

    // Free resources
    cudaFree(cdt.mem_dt.d_text);
    cudaFree(cdt.mem_dt.d_tokens);
    cudaFree(cdt.bpe_dt.freq_mat);
    cudaFree(cdt.bpe_dt.tok2bp1);
    cudaFree(cdt.bpe_dt.tok2bp2);
    delete[] h_tokens;

    return cd;
}