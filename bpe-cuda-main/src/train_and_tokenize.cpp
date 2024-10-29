#include "headers/CommonData.h"
#include "headers/bpe_seq.h"
#include <string>
#include <boost/regex.hpp>

std::vector<int> cleanup_na_tokens(const std::vector<int> &text_tokens) {
    std::vector<int> cleaned;
    std::copy_if(text_tokens.begin(), text_tokens.end(), std::back_inserter(cleaned),
                 [](int token) { return token != NA_TOKEN; });
    return cleaned;
}

std::vector<int> find_bp_indexes(const std::vector<int> &text_tokens, std::pair<int, int> bp) {
    std::vector<int> indexes;
    for (size_t i = 0; i < text_tokens.size() - 1; ++i) {
        if (text_tokens[i] == bp.first && text_tokens[i + 1] == bp.second) {
            indexes.push_back(i);
        }
    }
    return indexes;
}

std::vector<int> prep_str_for_tokenization(const std::string &text) {
    std::vector<int> tokens;
    for (char c: text) {
        tokens.push_back(static_cast<int>(static_cast<unsigned char>(c)));
    }
    return tokens;
}

std::pair<std::vector<int>, int> replace_bp_with_token(std::vector<int> &text_tokens, std::pair<int, int> bp, int tok) {
    auto indexes = find_bp_indexes(text_tokens, bp);
    for (auto i: indexes) {
        text_tokens[i] = tok;
        text_tokens[i + 1] = NA_TOKEN;
    }
    return std::make_pair(text_tokens, indexes.size());
}

std::string format_tokenized_text(const std::vector<int> &tokenized_text, char delimiter) {
    std::string formatted_text;
    for (size_t i = 0; i < tokenized_text.size(); ++i) {
        if (i > 0) formatted_text += delimiter;
        formatted_text += std::to_string(tokenized_text[i]);
    }
    return formatted_text;
}

int tokenize_phrase(std::vector<int> &text_tokens, CommonData &cd) {
    auto sorted_items = cd.get_sorted_tok2bp_items();
    int updates = 0;
    for (const auto &item: sorted_items) {
        int tok_id = item.first;
        auto byte_pair = item.second;
        auto res = replace_bp_with_token(text_tokens, byte_pair, tok_id);
        text_tokens = res.first;
        updates += res.second;
    }
    return updates;
}

std::vector<int> tokenize_phrase_mgr(std::vector<int> &text_tokens, CommonData &cd) {
    int i = 0;
    for (; i < TOK_MAX_ITERS; i++) {
        if (tokenize_phrase(text_tokens, cd) == 0)
            break;
    }
    text_tokens = cleanup_na_tokens(text_tokens);
#if(DEBUG == true)
    std::cout << "\nresult: iters:"<<i<<"\n";
    for (auto r: text_tokens) std::cout << r << ' ';
#endif

    return text_tokens;
}

std::vector<std::string> split(const std::string &input_text) {
    boost::regex regexPattern(PATTERN);

    boost::sregex_token_iterator iter(input_text.begin(), input_text.end(), regexPattern);
    boost::sregex_token_iterator end;

    std::vector<std::string> tokens;
    for (; iter != end; ++iter) {
        // Avoid adding empty strings that result from consecutive delimiters
        if (!iter->str().empty()) {
            tokens.push_back(*iter);
        }
    }

    return tokens;
}

/********************************************************************************/
std::pair<std::string, int> tokenize_text(const std::string &phrases, CommonData &cd) {
#if(DEBUG == true)
    std::cout<<" "<<phrases.length();
#endif
    std::vector<int> text_tokens = prep_str_for_tokenization(phrases);
    tokenize_phrase_mgr(text_tokens, cd);
    return std::make_pair(format_tokenized_text(text_tokens), text_tokens.size());
}

void train_tokenizer_on_phrase(std::vector<int> &text_tokens, CommonData &cd) {
    int i = 0;
    for (; i < TOK_MAX_ITERS; ++i) {
        cd.calc_byte_pair_frequencies(text_tokens);
        int token_id = 0;
        std::pair<int, int> max_freq_bp;

        if (cd.has_new_byte_pair_counts()) {
            max_freq_bp = cd.find_max_freq_token();
            token_id = cd.assign_new_token(max_freq_bp);
            cd.reset_byte_pair_counts();
        }

        if (token_id == 0) {
            break;
        }

        text_tokens = cleanup_na_tokens(replace_bp_with_token(text_tokens, max_freq_bp, token_id).first);
    }
#if DEBUG == true
    std::cout<<"Iters: "<<i;
#endif
    cd.training_iters=i;
}


CommonData train_tokenizer_seq(const std::string &input_text) {
    CommonData common_data = CommonData();
    auto chunks = split(input_text);
    for (const auto &s: chunks) {
        std::vector<int> text_tokens = prep_str_for_tokenization(s);
        train_tokenizer_on_phrase(text_tokens, common_data);
    }
    return common_data;
}


