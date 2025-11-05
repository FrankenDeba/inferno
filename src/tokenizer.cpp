#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <nlohmann/json.hpp>

#include "Config.h"

using namespace std;
using json = nlohmann::json;

vector<float> emb_weight = {0.15f, 0.25f, 0.35f, 0.45f};

struct TokenEntry {
    string token;
    int id;
};

Sizes model_sizes(4, 16, 6, 8, 8, 100, 1024);

int dim = model_sizes.dim;
int vocab_size = model_sizes.vocab_size;



vector<string> load_tokens(const string &text) {
    istringstream stream(text);

    vector<string> tokens;
    string token;

    while (stream >> token) {
        tokens.push_back(token);
    }

    // for (const auto &t : tokens) {
    //     cout << t << endl;
    // }

    return tokens;
}

vector<TokenEntry>
 load_vocab(const string &filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: could not find the vocabulary file: " << filename << endl;

        return vector<TokenEntry>{};
    }

    json words;

    file >> words;

    if (!words.is_object()) {
        std::cerr << "Expected object of token->id\n";
        return {};
    }

    vector<TokenEntry> vocab;
    vocab.reserve(words.size());

    for (auto& [token, id]: words.items()) {
        // cout << token << endl;
        TokenEntry entry;

        entry.token = token;
        entry.id = id.get<int>();
        vocab.push_back(entry);
    }

    return vocab;
}

vector<vector<float>> tokenize(const string &words) {
    vector<string> tokens = load_tokens(words);
    vector<TokenEntry> vocab = load_vocab("vocab.json");

    vector<vector<float>> embeddings;

    for (const auto &t : tokens) {
        // cout << "Token: " << t << endl;

         for (const auto &v : vocab) {
            if (v.token == t) {
                // cout << "Found token id: " << v.id << " " << t << endl;
                vector<float> emb(dim, 0.0f);
                for (int i = 0; i < dim; i++) {
                    emb[i] = emb_weight[i % emb_weight.size()] * v.id / 100.0f;
                }
                embeddings.push_back(emb);
            }
    }
    }

    // for (const auto &v : vocab) {
    //     cout << "Vocab: " << v.token << " " << v.id << endl;
    // }

    return embeddings;
}

int main() {
    vector<vector<float>> embeddings = tokenize("This is a sample text for tokenization");

    for (const auto& emb: embeddings) {
        for (const auto& val: emb) {
            cout << val << " ";
        }
        cout << endl;
    }


    return 0;
}

