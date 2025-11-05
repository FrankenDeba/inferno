#include <bits/stdc++.h>

using namespace std;

int main() {
    int dim = 4;
    int vocab = 10;

    vector<vector<float>> W(vocab, vector<float>(dim));

    for (auto &row : W)
        for (auto &val : row) val = ((rand() % 200) - 100) / 500.0f;

    int token_id = 5;

    vector<float> x = W[token_id];

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