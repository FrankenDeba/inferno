#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

using namespace std;

struct TokenEntry
{
    string token;
    int id;
};

// Declare your functions here
vector<TokenEntry> load_vocab(const string &filename);
vector<string> load_tokens(const string &text);
vector<vector<float>> encode(const string &text, const int &dim);
string decode(const int &token_id, const vector<TokenEntry> &vocab);

#endif
