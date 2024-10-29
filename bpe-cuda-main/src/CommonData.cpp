#include "src/headers/CommonData.h"
#include "src/headers/bpe_seq.h"

bool CommonData::has_new_byte_pair_counts() {
    return !bp_cnt.empty();
}

std::vector<std::pair<int, std::pair<int, int>>> CommonData::get_sorted_tok2bp_items(bool reverse) {
    std::vector<std::pair<int, std::pair<int, int>>> items(tok2bp.begin(), tok2bp.end());
    if (reverse) {
        std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    } else {
        std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    }
    return items;
}

int CommonData::get_new_id() {
    return TOKEN_START_VAL + tok2bp.size() + 1;
}

bool CommonData::has_token_id(std::pair<int, int> byte_pair) {
    return tokenized_byte_pairs.find(byte_pair) != tokenized_byte_pairs.end();
}

void CommonData::reset_byte_pair_counts() {
    bp_cnt.clear();
}

int CommonData::assign_new_token(std::pair<int, int> byte_pair) {
    if (has_token_id(byte_pair) || byte_pair == INVALID_PAIR) {
        return 0;
    }

    int token_id = get_new_id();
    tok2bp[token_id] = byte_pair;
    tokenized_byte_pairs.insert(byte_pair);
    return token_id;
}

void CommonData::calc_byte_pair_frequencies(const std::vector<int> &text_tokens) {
    for (size_t i = 0; i < text_tokens.size() - 1; ++i) {
        auto bp = std::make_pair(text_tokens[i], text_tokens[i + 1]);
        bp_cnt[bp] = bp_cnt.find(bp) != bp_cnt.end() ? bp_cnt[bp] + 1 : 1;
    }
}

std::pair<int, int> CommonData::find_max_freq_token() {
    // todo CUDA sort function and then get the top
    auto max_iter = std::max_element(
            bp_cnt.begin(),
            bp_cnt.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
#if(DEBUG == true)
    std::cout << "Max freq: " << max_iter->first.first << ", " << max_iter->first.second << " - "
              << max_iter->second << std::endl;
#endif
    if (max_iter->second > CONSIDERATION_THRESH)
        return max_iter->first;
    return INVALID_PAIR;
}
