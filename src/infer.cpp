#include <bits/stdc++.h>
#include "Config.h"

using namespace std;

Sizes model_sizes(4, 16, 6, 8, 8, 10, 1024);

int dim = model_sizes.dim;
int vocab = model_sizes.vocab_size;

int main() {

    vector<vector<float>> W(vocab, vector<float>(dim));

    for (auto &row : W)
        for (auto &val : row) val = ((rand() % 200) - 100) / 500.0f;

    int token_id = 5;


    vector<float> x = W[token_id];

    // cout << "Input token embedding: " << x << endl;

    for (int i = 0; i < x.size(); i++) {
        cout << x[i] << "  ";
    }

    vector<float> logits(vocab, 0.0f);

    for (int i = 0; i < vocab; i++) {
        for (int j = 0; j < dim; j++) {
            logits[i] += W[i][j] * x[j];
        }
    }

    int next = max_element(logits.begin(), logits.end()) - logits.begin();
    cout << "Next token predicted is: " << next << endl;

    return 0;
}