#include <chrono>
#include <sstream>
#include "headers/bpe_seq.h"
#include "headers/cuda_bpe.h"
#include "headers/CommonData.h"

double rounded_time(long time) {
    // gives denominator to divide to get time in seconds
    auto denominator = std::chrono::high_resolution_clock::duration::period::den;
    return (((double) time) / ((double) denominator))*1000; // convert to milli secs
}


std::pair<std::string, std::string> bpe_main(std::string input_text, bool parallel) {

#if DEBUG == true
    std::cout << "\ninput string h_text_length: " << input_text.length();
#endif

    CommonData cd;

    long total_time, mem_cpy_to_cuda = 0, cuda_train_time = 0, copy_from_cuda = 0;

    auto start = std::chrono::high_resolution_clock::now();
    if (parallel) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cuda_data_t cdt = init_and_copy_data_to_cuda(input_text);
        auto t1 = std::chrono::high_resolution_clock::now();
        cu_train_tokenizer(cdt);
        auto t2 = std::chrono::high_resolution_clock::now();
        cd = bring_data_from_cuda(cdt);
        auto t3 = std::chrono::high_resolution_clock::now();

        mem_cpy_to_cuda = (t1 - t0).count();
        cuda_train_time = (t2 - t1).count();
        copy_from_cuda = (t3 - t2).count();

    } else {
        cd = train_tokenizer_seq(input_text);
    }
    auto end = std::chrono::high_resolution_clock::now();
    total_time = (end - start).count();

    auto tok_res = tokenize_text(input_text, cd);

#if TESTING == true
    std::cout << "\noutput tokens: " << tok_res.second;


    std::string result = de_tokenize(tok_res.first, cd);
    std::cout << "\nde-tokenized result:" << result << std::endl;
#endif
    std::ostringstream metrics;
    metrics << "\"use_cuda\":" << (parallel ? "true" : "false")
            << ",\"input_tokens\":" << input_text.length()
            << ",\"output_tokens\":" << tok_res.second
            << ",\"training_iters\":" << cd.training_iters
            << ",\"total_time\":" << rounded_time(total_time)
            << ",\"mem_cpy_to_cuda\":" << rounded_time(mem_cpy_to_cuda)
            << ",\"cuda_train_time\":" << rounded_time(cuda_train_time)
            << ",\"copy_from_cuda\":" << rounded_time(copy_from_cuda);

    std::cout << metrics.str();
    return std::make_pair(tok_res.first, metrics.str());
}