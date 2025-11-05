#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

struct TokenEntry {
    std::string token;
    int id;
};

// Declare your functions here
std::vector<TokenEntry> load_vocab(const std::string &filename);
std::vector<std::string> load_tokens(const std::string &text);
std::vector<std::vector<float>> tokenize(const std::string &text, const int &dim);

#endif
