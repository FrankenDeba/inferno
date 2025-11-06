#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <tuple>

#include "Config.h"

using namespace std;

// Declare your functions here

tuple<vector<vector<float>>, vector<float>> initialize_wts(const Sizes &model_sizes);
vector<float> forward_pass(const Sizes &model_sizes, const vector<float> &embeddings);
tuple<float, int> predict(const Sizes &model_sizes, const vector<float> &embeddings);


#endif
