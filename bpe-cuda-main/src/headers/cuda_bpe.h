#ifndef BPE_CUDA_CUDA_BPE_H
#define BPE_CUDA_CUDA_BPE_H

#include <cuda_runtime.h>
#include <string>
#include <vector>

/***** Constants and Structs *****/
#define MAX_TOKENS 1000
#define MAX_THREADS 1024
#define NUM_THREADS(N)(MAX_THREADS<N?MAX_THREADS: N)
#define NUM_BLOCKS(N, num_t)(((int)(N/num_t))+ ((N%num_t)!=0))
#define FREQ_MAT_SIZE MAX_TOKENS*MAX_TOKENS // Example maximum, adjust as needed

typedef struct cu_bpe_core_logic_inputs {
    char *d_text;
    int *d_tokens;

    int inp_text_length;
    int tokens_length_h;
} cu_mem_t;

typedef struct cu_bpe_core_logic_data {
    int *tok2bp1, *tok2bp2;
    int *freq_mat;
} cu_bpe_t;

typedef struct cu_bpe_temp_processing_data {
    int bp1 = -1, bp2 = -1, max_freq_val = -1, max_freq_idx = -1;
    int tok_id_if_exists = -1, latest_tok_id = -1;
    int num_replacements = -1;
    int tokens_length_d = -1;
    bool convergence_reached = false, max_val_valid = false;
    int temp = 1;
} cu_bpe_processing_t;

typedef struct {
    cu_mem_t mem_dt;
    cu_bpe_t bpe_dt;
    cu_bpe_processing_t *proc_dt;
    int train_iters=0;
} cuda_data_t;

/***** Function declarations *****/
cuda_data_t init_and_copy_data_to_cuda(const std::string &text);

// bpe logic functions in cuda
void cu_reset_byte_pair_counts(cuda_data_t cdt);

void cu_find_max_freq_token(cuda_data_t &cdt);

void cu_train_tokenizer(cuda_data_t &cdt);

void cu_cleanup_na_tokens(cuda_data_t &cdt);
/*****************************************************************/
///// CUDA kernels
/*****************************************************************/
__global__ void cukl_char_to_int(cuda_data_t cdt);

__global__ void cukl_cal_byte_pair_freq(cuda_data_t cdt);

__global__ void cukl_check_for_existing_token(cuda_data_t cdt);

__global__ void cukl_assign_new_token(cuda_data_t cdt);

__global__ void cukl_replace_bp_with_token(cuda_data_t cdt);

__global__ void cukl_replace_bp_with_token_st(cuda_data_t cdt);

__global__ void cukl_cleanup_na_tokens(cuda_data_t cdt);

__global__ void cukl_find_max_freq(cuda_data_t cdt, int n_items);

__global__ void cukl_set_max_value(cuda_data_t cdt);

__global__ void cukl_find_max_freq_st(cuda_data_t cdt);

#endif //BPE_CUDA_CUDA_BPE_H
