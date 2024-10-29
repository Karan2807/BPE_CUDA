#include "headers/cuda_bpe.h"
#include "headers/CommonData.h"
#include "src/headers/bpe_seq.h"

#define ENSURE_SINGLE_THREADED_EXEC if (threadIdx.x != 0) return

__device__ int get_new_id(cuda_data_t cdt) {
    return TOKEN_START_VAL + cdt.proc_dt->latest_tok_id + 1;
}

//#define BP_IDX(bp1, bp2) (bp1 * MAX_TOKENS + bp2)
#define BP_IDX(arr, idx) ((arr[idx]*MAX_TOKENS)+arr[idx+1])


__global__ void cukl_char_to_int(cuda_data_t cdt) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < cdt.mem_dt.inp_text_length) {
        cdt.mem_dt.d_tokens[index] = static_cast<int>(static_cast<unsigned char>(cdt.mem_dt.d_text[index]));
    }
}

__global__ void cukl_cal_byte_pair_freq(cuda_data_t cdt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cdt.mem_dt.inp_text_length - 1) return;

    int bpIndex = BP_IDX(cdt.mem_dt.d_tokens, idx);
    cdt.bpe_dt.freq_mat[bpIndex] = 0;
    __syncthreads();

    /*todo - use shared memory to store bpidx and freq in 2 halves of an array
      and then do atomic updates at the end from tid 0*/
    atomicAdd(&(cdt.bpe_dt.freq_mat[bpIndex]), 1); // add up frequencies
}

__global__ void cukl_check_for_existing_token(cuda_data_t cdt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x == 0) cdt.proc_dt->tok_id_if_exists = -1;
    __syncthreads();

    if (idx < MAX_TOKENS) {
        if (cdt.bpe_dt.tok2bp1[idx] == cdt.proc_dt->bp1 && cdt.bpe_dt.tok2bp2[idx] == cdt.proc_dt->bp2) {
            cdt.proc_dt->tok_id_if_exists = idx;
        }
    }
}

__global__ void cukl_assign_new_token(cuda_data_t cdt) {
    ENSURE_SINGLE_THREADED_EXEC;
    cdt.proc_dt->tok_id_if_exists = -1;

    if (cdt.proc_dt->max_val_valid && cdt.proc_dt->latest_tok_id < MAX_TOKENS) {
        cukl_check_for_existing_token<<<NUM_BLOCKS(MAX_TOKENS, NUM_THREADS(MAX_TOKENS)),
        NUM_THREADS(MAX_TOKENS) >>>(cdt);

        if (cdt.proc_dt->tok_id_if_exists == -1) {
            int token_id = get_new_id(cdt);
            cdt.bpe_dt.tok2bp1[token_id] = cdt.proc_dt->bp1;
            cdt.bpe_dt.tok2bp2[token_id] = cdt.proc_dt->bp2;
            cdt.proc_dt->latest_tok_id += 1;
        }
    }

    if ((cdt.proc_dt->latest_tok_id == MAX_TOKENS) || !cdt.proc_dt->max_val_valid) {
        cdt.proc_dt->convergence_reached = true;
    }
}



__global__ void cukl_replace_bp_with_token(cuda_data_t cdt) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int replacement = 0;
    if (threadIdx.x == 0) cdt.proc_dt->num_replacements = 0;

    if (index < (cdt.mem_dt.tokens_length_h - 1)) {
        if ((cdt.mem_dt.d_tokens[index] == cdt.proc_dt->bp1) && (cdt.mem_dt.d_tokens[index + 1] == cdt.proc_dt->bp2)) {
            cdt.mem_dt.d_tokens[index] = cdt.proc_dt->latest_tok_id;
            cdt.mem_dt.d_tokens[index + 1] = NA_TOKEN;
            replacement = 1;
        }
    }

    __syncthreads();

    if (replacement > 0) { atomicAdd(&cdt.proc_dt->num_replacements, 1); }
}

__global__ void cukl_cleanup_na_tokens(cuda_data_t cdt) { //single threaded :(
    ENSURE_SINGLE_THREADED_EXEC;

    int src = 0, dst = 0, n = cdt.mem_dt.tokens_length_h;
    while (src < n) {
        while (src < n && cdt.mem_dt.d_tokens[src] == NA_TOKEN) {
            src += 1;
        }
        if (dst >= n || src >= n) break;
        cdt.mem_dt.d_tokens[dst] = cdt.mem_dt.d_tokens[src];
        src++;
        dst++;
    }
    cdt.proc_dt->tokens_length_d = dst;
}

__global__ void cukl_find_max_freq_st(cuda_data_t cdt) {
    ENSURE_SINGLE_THREADED_EXEC;
    int max_val = -1;
    int max_idx = -1;
    for (int i = 0; i < FREQ_MAT_SIZE; i++) {
        max_idx = cdt.bpe_dt.freq_mat[i] > max_val ? i : max_idx;
        max_val = cdt.bpe_dt.freq_mat[i] > max_val ? cdt.bpe_dt.freq_mat[i] : max_val;
    }
    cdt.proc_dt->max_freq_val = max_val;
    cdt.proc_dt->max_freq_idx = max_idx;
}

__global__ void cukl_find_max_freq(cuda_data_t cdt, int n_items) {
    // at-least 100 items per thread
    extern __shared__ int max_data_arr[];
    // max_data_arr size = blockDim (num threads)
    // so make sure to have not too many threads
    // first half for max val, second half for max idx
    int *max_val_arr = max_data_arr;
    int *max_idx_arr = &max_data_arr[blockDim.x];

    int local_max_val = -1;
    int local_max_idx = -1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    max_val_arr[tid] = -1;
    max_idx_arr[tid] = -1;

    if ((idx + n_items) >= FREQ_MAT_SIZE) return;

    // find max value among items given to the thread
    for (int i = 0; i < n_items; i++) {
        int a0 = cdt.bpe_dt.freq_mat[idx + i];
        local_max_idx = a0 > local_max_val ? idx + i : local_max_idx;
        local_max_val = a0 > local_max_val ? a0 : local_max_val;
    }
    max_val_arr[tid] = local_max_val;
    max_idx_arr[tid] = local_max_idx;

    __syncthreads();

    if (tid == 0) {
        // find the block level max value
        int mxv = -1, mxidx = -1;
        for (int i = 0; i < blockDim.x; i++) {
            mxidx = max_val_arr[i] > mxv ? max_idx_arr[i] : mxidx;
            mxv = max_val_arr[i] > mxv ? max_val_arr[i] : mxv;
        }
        // atomic max using max value from the block
        atomicMax(&(cdt.proc_dt->max_freq_idx), mxidx);
        atomicMax(&(cdt.proc_dt->max_freq_val), mxv);
    }
}

__global__ void cukl_set_max_value(cuda_data_t cdt) {
    ENSURE_SINGLE_THREADED_EXEC;
    int result_index = cdt.proc_dt->max_freq_idx;
    int max_value = cdt.proc_dt->max_freq_val;
    int bp1 = (int) (result_index / MAX_TOKENS);
    int bp2 = (int) (result_index % MAX_TOKENS);
    if (max_value > CONSIDERATION_THRESH) {
        cdt.proc_dt->max_val_valid = true;
        cdt.proc_dt->bp1 = bp1;
        cdt.proc_dt->bp2 = bp2;
    } else {
        cdt.proc_dt->bp1 = cdt.proc_dt->bp2 = -1;
        cdt.proc_dt->max_val_valid = false;
    }

}

/// not used
__global__ void cukl_replace_bp_with_token_st(cuda_data_t cdt) { // for debugging end error isolation

    int replacements = 0;
    for (int index = 0; index < cdt.mem_dt.inp_text_length - 1; index++) {
        if ((cdt.mem_dt.d_tokens[index] == cdt.proc_dt->bp1) && (cdt.mem_dt.d_tokens[index + 1] == cdt.proc_dt->bp2)) {
            cdt.mem_dt.d_tokens[index] = cdt.proc_dt->latest_tok_id;
            cdt.mem_dt.d_tokens[index + 1] = NA_TOKEN;
            replacements += 1;
        }
    }
    if (replacements > 0) { atomicAdd(&cdt.proc_dt->num_replacements, replacements); }
}