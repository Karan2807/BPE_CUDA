#ifndef BPE_CUDA_BPE_SEQ_H
#define BPE_CUDA_BPE_SEQ_H

#include <iostream>
#include "CommonData.h"

#define NA_TOKEN MAX_TOKENS+5 //std::numeric_limits<int>::max()
#define TOKENIZED_TEXT_DELIMITER ','

#define DEBUG false
#define TESTING false


//#define PATTERN "'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
#define PATTERN ".*?[.!?]\\s*"
#define TOK_MAX_ITERS 1000
#define INVALID_PAIR std::make_pair(-1,-1)

/**********************************************************************************************/

std::vector<int> find_bp_indexes(const std::vector<int> &text_tokens, std::pair<int, int> bp);

void train_tokenizer_on_phrase(std::vector<int> &text_tokens, CommonData &cd);

std::vector<int> prep_str_for_tokenization(const std::string &text);

std::vector<std::string> split(const std::string &input_text);

std::vector<int> cleanup_na_tokens(const std::vector<int> &text_tokens);

/**********************************************************************************************/
std::string format_tokenized_text(const std::vector<int> &tokenized_text, char delimiter = TOKENIZED_TEXT_DELIMITER);

std::vector<int> tokenize_phrase(const std::string &phrase, CommonData &cd);

std::pair<std::string, int> tokenize_text(const std::string &phrases, CommonData &cd);

CommonData train_tokenizer_seq(const std::string &input_text);

/**********************************************************************************************/
std::vector<int> unformat_tokenized_text(const std::string &tokenized_text);

std::vector<int> de_tokenize_phrase(const std::string &tokenized_text_phrase);

std::string charcodes_to_string(const std::vector<int> &char_codes);

std::string de_tokenize(const std::string &tokenized_text, CommonData common_data);

/**********************************************************************************************/
#endif //BPE_CUDA_BPE_SEQ_H
