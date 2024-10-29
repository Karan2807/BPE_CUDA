#include "headers/bpe_seq.h"
#include "headers/cuda_bpe.h"
#include <boost/regex.hpp>

////// for debugging ////////

#include <vector>
void print_byte_pair_freqs(cuda_data_t cdt) {
    int length = FREQ_MAT_SIZE;
    int check[length];
    cudaMemcpy(check, cdt.bpe_dt.freq_mat, length * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "\n\nfreq_mat:";
    for (int i = 0; i < length; i++)
        if (check[i] >1) {
            int bp1 = (int) (i / MAX_TOKENS);
            int bp2 = (int) (i % MAX_TOKENS);
            std::cout << "," << i << "(" << bp1 << "," << bp2 << ")" << ":" << check[i];
        }
}

void print_max_fre_val(cuda_data_t cdt) {
    cu_bpe_processing_t bp_proc_local;
    cudaMemcpy(&bp_proc_local, cdt.proc_dt,
               sizeof(cu_bpe_processing_t), cudaMemcpyDeviceToHost);
    std::cout << "\n max_freq_info: " << bp_proc_local.max_freq_idx << "," << bp_proc_local.max_freq_val << ","
              << bp_proc_local.max_val_valid;
    std::cout << "\n bp info " << bp_proc_local.bp1 << "," << bp_proc_local.bp2;
}


/////////////////////////////////////////////////////////////////////////////////
//////////////////////// Core logic//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

void cu_reset_byte_pair_counts(cuda_data_t cdt) {
    cudaMemset(cdt.bpe_dt.freq_mat, 0, FREQ_MAT_SIZE * sizeof(int));
}

void cu_find_max_freq_token(cuda_data_t &cdt) {

    int num_elems_per_thread = ceil(MAX_TOKENS / 10);
    int threads = MAX_THREADS; //ceil(FREQ_MAT_SIZE / num_elems_per_thread);

    int blocks = (int) (FREQ_MAT_SIZE / (num_elems_per_thread * threads)) +
                 ((FREQ_MAT_SIZE % (num_elems_per_thread * threads)) != 0);

    int sh_mem_size = threads * 2 * (int) sizeof(int);

#if DEBUG == true
    //    std::cout << "\nmeta: " << num_elems_per_thread
    //              << "," << blocks << "," << threads << ',' << sh_mem_size;
#endif
//    cukl_find_max_freq<<<blocks, threads, sh_mem_size>>>(cdt, num_elems_per_thread);

    cukl_find_max_freq_st<<<1, 1>>>(cdt);
    cukl_set_max_value<<<1, 1>>>(cdt);
}


void cu_cleanup_na_tokens(cuda_data_t &cdt) {
#if DEBUG == true
    int mem_size = cdt.mem_dt.tokens_length_h;
    int length = mem_size;
    int check[length];
    cudaDeviceSynchronize();
    cudaMemcpy(check, cdt.mem_dt.d_tokens, length * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\nd_tokens_tmp:\n";
    for (int i = 0; i < length; i++)std::cout << "," << check[i];
    std::cout << "\nd_tokens_tmp end \n";
#endif
    cukl_cleanup_na_tokens<<<1, 1>>>(cdt);
}


bool check_for_convergence(cuda_data_t &cdt) {
    cu_bpe_processing_t bp_proc_local;

    cudaMemcpy(&bp_proc_local, cdt.proc_dt,
               sizeof(cu_bpe_processing_t), cudaMemcpyDeviceToHost);

    cdt.mem_dt.tokens_length_h = bp_proc_local.tokens_length_d;

#if DEBUG == true
    std::cout << "\nbp1:" << bp_proc_local.bp1 << ", bp2:" << bp_proc_local.bp2
              << ", temp:" << bp_proc_local.temp
              << ", max_freq_val:" << bp_proc_local.max_freq_val
              << ", max_freq_idx:" << bp_proc_local.max_freq_idx
              << ", max_val_valid:" << bp_proc_local.max_val_valid
              << ", convergence_reached:" << bp_proc_local.convergence_reached
              << ", latest_tok_id:" << bp_proc_local.latest_tok_id
              << ", num_replacements:" << bp_proc_local.num_replacements
              << ", tokens_length_d:" << bp_proc_local.tokens_length_d
              << ", tok_id_if_exists:" << bp_proc_local.tok_id_if_exists;
#endif
    return bp_proc_local.convergence_reached;
}

void cu_train_tokenizer_on_phrase(cuda_data_t &cdt) {
    // keep num iterations a high number, so that it will converge and break loop as soon as it's done
    int i = 0;
    for (; i < TOK_MAX_ITERS; i++) {
        int n = cdt.mem_dt.tokens_length_h;

        cukl_cal_byte_pair_freq<<<NUM_BLOCKS(n, NUM_THREADS(n)), NUM_THREADS(n)>>>(cdt);
        cu_find_max_freq_token(cdt);

        cukl_assign_new_token<<<1, 1>>>(cdt);//TBD, make sure to initialize all asccii values ?
        cu_reset_byte_pair_counts(cdt); // fully device code

        cukl_replace_bp_with_token<<<NUM_BLOCKS(n, NUM_THREADS(n)), NUM_THREADS(n)>>>(cdt);

        cu_cleanup_na_tokens(cdt);
        bool converged = check_for_convergence(cdt);

        if (converged)
            break;
    }
#if DEBUG == true
    std::cout << "\ntotal iterations:" << i << ' ';
#endif
    cdt.train_iters=i;
}


void cu_prep_str_for_tokenization(cuda_data_t cdt) {
    cukl_char_to_int<<<512, (int) ceil(cdt.mem_dt.inp_text_length / 512.0)>>>(cdt);
}


void cu_train_tokenizer(cuda_data_t &cdt) {
    cu_prep_str_for_tokenization(cdt);
    cu_train_tokenizer_on_phrase(cdt);
}