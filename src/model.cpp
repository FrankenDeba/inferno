#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>

#include "Config.h"
#include "tokenizer.h"

using namespace std;

tuple<vector<vector<float>>, vector<float>> initialize_wts(const Sizes &model_sizes)
{
    int dim = model_sizes.dim;
    int vocab_size = model_sizes.vocab_size;
    vector<vector<float>> weights(vocab_size, vector<float>(dim));
    vector<float> biases(dim);

    for (int i = 0; i < vocab_size; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            // random float between -0.05 and +0.05
            weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) / 10.0f;
        }
    }

    for (int i = 0; i < dim; i++)
    {
        biases[i] = ((float)rand() / RAND_MAX - 0.5f) / 10.0f;
    }

    return {weights, biases};
}

vector<float> forward_pass(const Sizes &model_sizes, const vector<float> &embeddings, const vector<vector<float>> &w, const vector<float> &b) {

    vector<float> logits(model_sizes.vocab_size, 0.0f);

    for (int i = 0; i < model_sizes.vocab_size; i++)
    {
        for (int j = 0; j < model_sizes.dim; j++)
        {
            logits[i] += w[i][j] * embeddings[j];
        }
    }

    return logits;
}

tuple<float, int> predict(const Sizes &model_sizes, const vector<float> &embeddings, const vector<vector<float>> &w, const vector<float> &b)
{
    auto logits = forward_pass(model_sizes, embeddings, w, b);

    int max_logit_index = distance(logits.begin(), max_element(logits.begin(), logits.end()));

    float max_logit = logits[max_logit_index];

    return {max_logit, max_logit_index};
}