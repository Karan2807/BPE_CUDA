import regex as re
from builtins import *
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set

NA_TOKEN = float('inf')
TOKENIZED_TEXT_DELIMITER = ','
TOKEN_START_VAL = 256  # 0-255 ascii
SENT_DELIM = '<<EOS>>'
DEBUG = False


@dataclass
class CommonData:
    tok2bp: Dict[int, Tuple[int]] = field(default_factory=dict)

    # required only during processing
    tokenized_byte_pairs: Set[int] = field(default_factory=set)
    bp_cnt: Dict[Tuple[int], int] = field(default_factory=dict)

    def has_new_byte_pair_counts(self):
        return self.bp_cnt != {}

    def get_sorted_tok2bp_items(self, reverse=False):
        return sorted(self.tok2bp.items(), key=lambda item: item[0], reverse=reverse)

    def get_new_id(self):
        return TOKEN_START_VAL + len(self.tok2bp) + 1

    def has_token_id(self, byte_pair):
        return byte_pair in self.tokenized_byte_pairs

    def reset_byte_pair_counts(self, ):
        self.bp_cnt = {}

    def assign_new_token(self, byte_pair):
        if byte_pair is None or self.has_token_id(byte_pair):
            return

        token_id = self.get_new_id()
        self.tok2bp[token_id] = byte_pair
        self.tokenized_byte_pairs.add(byte_pair)
        return token_id

    def calc_byte_pair_frequencies(self, text_tokens):
        for i in range(len(text_tokens) - 1):
            bp = text_tokens[i], text_tokens[i + 1]
            self.bp_cnt[bp] = self.bp_cnt.get(bp, 0) + 1

    def find_max_freq_token(self, ):
        items = sorted(self.bp_cnt.items(), key=lambda item: item[1], reverse=True)
        if items[0][1] > 1:
            max_freq = items[0]
            print("items:", items, "max_freq:", max_freq) if DEBUG else 0
            return max_freq[0]


def split(input_text):
    # pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    # return re.findall(re.compile(pattern), input_text)
    pattern = r"[^\n\r\.,]+[\n\r\.,]"
    return re.findall(re.compile(pattern), input_text)


def find_bp_indexes(text_tokens, bp):
    v1, v2 = bp
    v1_idxs = list(filter(lambda i: text_tokens[i] == v1, list(range(len(text_tokens) - 1))))
    return list(filter(lambda i: text_tokens[i + 1] == v2, v1_idxs))


def replace_bp_with_token(text_tokens, bp, tok, ret_cnt=False):
    bp_idxs = find_bp_indexes(text_tokens, bp)
    for i in reversed(bp_idxs):
        text_tokens[i] = tok
        text_tokens[i + 1] = NA_TOKEN
    if ret_cnt:
        return text_tokens, len(bp_idxs)
    return text_tokens


def cleanup_na_tokens(text_tokens):
    return list(filter(lambda i: i != NA_TOKEN, text_tokens))


def train_tokenizer_on_phrase(text_tokens, cd):
    iters = len(text_tokens)  # min(int(len(text_tokens) / 2), 10)
    cnt = 0
    for i in range(iters):
        print("\niter " + str(i + 1)) if DEBUG else 0
        cnt += 1
        cd.calc_byte_pair_frequencies(text_tokens)
        token_id, max_freq_bp = None, None

        if cd.has_new_byte_pair_counts():
            max_freq_bp = cd.find_max_freq_token()
            token_id = cd.assign_new_token(max_freq_bp)
            cd.reset_byte_pair_counts()

        if token_id is None:
            if max_freq_bp is None:
                print("All of the given text is completely tokenized!") if DEBUG else 0
                print("Tokenizer converged! breaking the loop") if DEBUG else 0
                break
            elif len(find_bp_indexes(text_tokens, max_freq_bp)) > 0:
                print("Max freq bp already has a token but not replaced !", max_freq_bp,
                      find_bp_indexes(text_tokens, max_freq_bp), cd) if DEBUG else 0
            else:
                print("Tokenizer converged! breaking the loop") if DEBUG else 0
                break
        text_tokens = cleanup_na_tokens(replace_bp_with_token(text_tokens, max_freq_bp, token_id))
        print("\nphrase:", text_tokens) if DEBUG else 0

    print(" iter:", cnt, end=' ') if DEBUG else 0
    return cd, text_tokens


def format_tokenized_text(tokenized_text: List[int], delimeter=TOKENIZED_TEXT_DELIMITER) -> str:
    return delimeter.join([str(t) for t in tokenized_text])


def prep_str_for_tokenization(text):
    return list(map(int, text.encode("utf-8")))


def train_tokenizer(input_text):
    common_data = CommonData()
    combined_res = []
    for s in split(input_text):
        # print("TEXT CHUNK:", s, end=' ')
        text_tokens = prep_str_for_tokenization(s)
        common_data, res = train_tokenizer_on_phrase(text_tokens, common_data)
        combined_res.extend(res)

    # print("\nafter training:", len(res), combined_res)
    common_data, res = train_tokenizer_on_phrase(combined_res, common_data)
    # print("\nafter training:", len(res))
    return common_data


###################################################################################################
def tokenize_split_phrase(phrase, cd):
    iters = 0
    while iters < 1000:
        updates = 0
        for tok_id, byte_pair in cd.get_sorted_tok2bp_items():
            result, cnt = replace_bp_with_token(phrase, byte_pair, tok_id, True)
            result = cleanup_na_tokens(result)
            iters += 1
            updates += cnt

        if updates == 0:
            break

    # print("\noriginal:", len(phrase), "result:", len(result))
    # print("Common data", cd)
    # return format_tokenized_text(result)
    return result


def tokenize_text(phrases, cd):
    split_phrases = split(phrases)
    fr = []
    for phrase in split_phrases:
        phrase = prep_str_for_tokenization(phrase)
        fr.extend(tokenize_split_phrase(phrase, cd))

    # print("\nPHINAL!", len(fr), fr)
    fr = tokenize_split_phrase(fr, cd)
    # print(len(fr), cd)
    return format_tokenized_text(fr), len(fr)


###################################################################################################
###################################################################################################

def charcodes_to_string(char_codes_arr: List[int]) -> str:
    text_bytes = bytes(char_codes_arr)
    return text_bytes.decode("utf-8", errors="replace")


def flatten_list(list_with_iterables):
    result = []
    for e in list_with_iterables:
        if isinstance(e, tuple) or isinstance(e, list):
            result.extend(e)
        else:
            result.append(e)
    return result


def replace_token_with_bp(text_tokens, tok, bp):
    tok_idxs = list(filter(lambda i: text_tokens[i] == tok, list(range(len(text_tokens)))))
    for i in tok_idxs:
        text_tokens[i] = bp

    return text_tokens, len(tok_idxs)


def unformat_tokenized_text(tokenized_text: str, delimeter=TOKENIZED_TEXT_DELIMITER) -> List[int]:
    return [int(t) for t in tokenized_text.split(delimeter)]


def de_tokenize_phrase(tokenized_text_phrase, common_data):
    result = unformat_tokenized_text(tokenized_text_phrase)
    for i in range(1000):
        total_updates = 0
        for item in common_data.get_sorted_tok2bp_items(reverse=True):
            token, bp = item
            result, updates = replace_token_with_bp(result, token, bp)
            result = flatten_list(result)
            total_updates += updates

        if total_updates == 0:
            break

    return result


def de_tokenize(tokenized_text, common_data):
    return charcodes_to_string(de_tokenize_phrase(tokenized_text, common_data))


###################################################################################################
###################################################################################################

def main(input_text):
    print("\nInput length:", len(input_text))
    common_data = train_tokenizer(input_text)
    tokenized_text, num_elems = tokenize_text(input_text, common_data)
    print("\nnum elements:", num_elems, " tokenized_text: ", tokenized_text)
    detok = de_tokenize(tokenized_text, common_data)
    print("de-tokenized text: ", detok)


#######################################################################################
#######################################################################################

input_text_string = """
In Python, while there's no specific representation for "integer infinity" because Python's integers are unbounded, you can use float('inf') for a conceptually similar purpose in situations where you're looking for a value that represents infinity. This is useful for algorithms where you need a value that is guaranteed to be greater than any other value you're working with. Even though float('inf') is a floating-point number, it can be used in contexts where an "infinite" integer value might be conceptually required, such as initializing variables for comparison in search algorithms.

Here's an example where you might use float('inf') as an "integer infinity" in an algorithm:
# Example: Find the minimum value in a list, with a fallback to "infinity" if the list is empty

# Let's say we have a list of integers
numbers = [10, 20, 3, 40, 5]

# Initialize minimum value as "infinity"
min_value = float('inf')

# Loop through each number in the list to find the minimum
for number in numbers:
    if number < min_value:
        min_value = number

# If the list was empty, min_value will remain as "infinity"
# Otherwise, it will be the minimum number found
if min_value == float('inf'):
    print("The list is empty or all values were infinity.")
else:
    print(f"The minimum value is: {min_value}")

# This approach can be particularly useful in algorithms where you need to compare values and want to start with an "infinite" value.
"""
main(input_text_string)
