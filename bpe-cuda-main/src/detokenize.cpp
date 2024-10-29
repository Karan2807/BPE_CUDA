#include "headers/CommonData.h"
#include "headers/bpe_seq.h"
#include <string>
#include <sstream> // For std::stringstream

int replace_token_with_bp(std::vector<int> &text_tokens, int tok, std::pair<int, int> bp) {
    //todo cuda kernel
    int updates = 0;
    for (auto it = text_tokens.begin(); it != text_tokens.end();) {
        if (*it == tok) {
            *it = bp.first;
            it = text_tokens.insert(it + 1, bp.second);
            updates += 1;
        }
        ++it;
    }
    return updates;
}

std::vector<int> unformat_tokenized_text(const std::string &tokenized_text) {
    std::vector<int> result;
    std::stringstream ss(tokenized_text);
    std::string item;
    while (std::getline(ss, item, TOKENIZED_TEXT_DELIMITER)) {
        result.push_back(std::stoi(item));
    }
    return result;
}

std::vector<int> de_tokenize_phrase(const std::string &tokenized_text_phrase, CommonData cd) {
    std::vector<int> text_tokens = unformat_tokenized_text(tokenized_text_phrase);
    auto reverse_sorted_items = cd.get_sorted_tok2bp_items(true);
#if DEBUG==true
    std::cout << "\ncdt: ";
    for (const auto &item: reverse_sorted_items) {
        std::cout << " " << item.first << ":(" << item.second.first << "," << item.second.second << ") ";
    }
#endif
    for (int i = 0; i < TOK_MAX_ITERS; i++) {
        int updates = 0;

        for (const auto &item: reverse_sorted_items) {
            updates += replace_token_with_bp(text_tokens, item.first, item.second);
        }

        if (updates == 0)
            break;
    }
    return text_tokens;
}

std::string charcodes_to_string(const std::vector<int> &char_codes) {
    std::string result;
    for (int code: char_codes) {
        result += static_cast<char>(code);
    }
    return result;
}

std::string de_tokenize(const std::string &tokenized_text, CommonData cd) {
    return charcodes_to_string(de_tokenize_phrase(tokenized_text, cd));
}

