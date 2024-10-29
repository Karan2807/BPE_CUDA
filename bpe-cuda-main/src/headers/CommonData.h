#ifndef BPE_CUDA_COMMONDATA_H
#define BPE_CUDA_COMMONDATA_H

#include <cstdlib>
#include <unordered_map>
#include <set>
#include <vector>
#include <algorithm>

#define TOKEN_START_VAL 256  // 0-255 ascii
#define CONSIDERATION_THRESH 1
#define  TIME_IN_MS(start, stop) ((double)(stop - start) * 1000.0 / CLOCKS_PER_SEC)
#include <iostream>
#include <unordered_map>
#include <utility> // For std::pair
#include "cuda_bpe.h"

#define  INVALID_PAIR std::make_pair(-1,-1)

// Define a custom hash function for std::pair<int, int>
struct pair_hash {
    template<class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ hash2; // Combine the two hash values. (You might want a better combination method.)
    }
};

class CommonData {

private:
    std::unordered_map<int, std::pair<int, int>> tok2bp;

    // required for processing
    std::set<std::pair<int, int>> tokenized_byte_pairs;
    std::unordered_map<std::pair<int, int>, int, pair_hash> bp_cnt;


public:
    int training_iters=0;
    CommonData() {}

    CommonData(std::unordered_map<int, std::pair<int, int>> tok2bpmap) {
        tok2bp = tok2bpmap;
    }

    ~CommonData() = default;

    bool has_new_byte_pair_counts();

    std::vector<std::pair<int, std::pair<int, int>>> get_sorted_tok2bp_items(bool reverse = false);

    int get_new_id();

    bool has_token_id(std::pair<int, int> byte_pair);

    void reset_byte_pair_counts();

    int assign_new_token(std::pair<int, int> byte_pair);

    void calc_byte_pair_frequencies(const std::vector<int> &text_tokens);

    std::pair<int, int> find_max_freq_token();
};

CommonData bring_data_from_cuda(cuda_data_t cdt);

#endif //BPE_CUDA_COMMONDATA_H
