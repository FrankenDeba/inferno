#include <bits/stdc++.h>
#include <tuple>

#include "Config.h"
#include "tokenizer.h"
#include "model.h"

using namespace std;
using Matrix = vector<vector<float>>;

Sizes model_sizes(4, 16, 6, 8, 8, 10, 1024);

int dim = model_sizes.dim;
int vocab = model_sizes.vocab_size;

tuple<Matrix, Matrix, Matrix> matmul(int seq_n, Matrix &X, Matrix &W_Q, Matrix &W_K, Matrix &W_V) {
// seq_n: number of tokens in the sequence
// dim: dimension of each token embedding
// X: input embeddings of shape (seq_n, dim)
// W_Q, W_K, W_V: weight matrices of shape (dim, dim)
// W_Q, W_K, W_V are the learned projection matrices for queries, keys, and values respectively
Matrix Q(seq_n, vector<float>(dim, 0.0f));
Matrix K(seq_n, vector<float>(dim, 0.0f));
Matrix V(seq_n, vector<float>(dim, 0.0f));

for (int i = 0; i < seq_n; i++) {
    for (int j = 0; j < dim; j++) {
        float sum = 0.0f;
        for (int k = 0; k < dim; k++) {
            // Sum is the dot product of row i of X and column j of W_Q
            // X[i][k] is the k-th element of the i-th token embedding
            // W_Q[k][j] is the k-th element of the j-th column of W_Q
            // Q[i][j] is the j-th element of the query vector for the i-th token

            Q[i][j] += X[i][k] * W_Q[k][j];
            K[i][j] += X[i][k] * W_K[k][j];
            V[i][j] += X[i][k] * W_V[k][j];
        }
    }
}

return {Q, K, V};
}

int main()
{

    // vector<vector<float>> W(vocab, vector<float>(dim));

    // for (auto &row : W)
    //     for (auto &val : row) val = ((rand() % 200) - 100) / 500.0f;

    // int token_id = 5;

    string input = "an animal";
    auto vocab = load_vocab("vocab.json");
    auto tokens = load_tokens(input);
    auto embeddings = encode(input, dim);

    auto [W, b] = initialize_wts(model_sizes);

    auto [max_logit, predicted_token] = predict(model_sizes, embeddings[0], W, b);

    cout << "Predicted token id: " << predicted_token << " with logit: " << max_logit << endl;

    cout << "predicted word: " << decode(predicted_token, vocab) << endl;

    cout << "Printing embeddings: \n";
    for (auto &emb : embeddings)
    {
        for (auto &val : emb)
        {
            cout << val << " ";
        }

        cout << "\n"
             << endl;
    }

    // vector<float> x = W[token_id];

    // // cout << "Input token embedding: " << x << endl;

    // for (int i = 0; i < x.size(); i++) {
    //     cout << x[i] << "  ";
    // }

    // vector<float> logits(vocab, 0.0f);

    // for (int i = 0; i < vocab; i++) {
    //     for (int j = 0; j < dim; j++) {
    //         logits[i] += W[i][j] * x[j];
    //     }
    // }

    // int next = max_element(logits.begin(), logits.end()) - logits.begin();
    // cout << "Next token predicted is: " << next << endl;

    return 0;
}