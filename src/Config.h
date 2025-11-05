// Config.h
#ifndef CONFIG_H
#define CONFIG_H

class Sizes {
public:
    int dim;          // transformer dimension
    int hidden_dim;   // for ffn layers
    int n_layers;     // number of layers
    int n_heads;      // number of query heads
    int n_kv_heads;   // number of key/value heads (can be < query heads)
    int vocab_size;   // vocabulary size, usually 256 (byte-level)
    int seq_len;      // max sequence length

    // Default constructor
    Sizes() 
        : dim(0), hidden_dim(0), n_layers(0), n_heads(0),
          n_kv_heads(0), vocab_size(0), seq_len(0) {}

    // Parameterized constructor
    Sizes(int dim, int hidden_dim, int n_layers, int n_heads, 
           int n_kv_heads, int vocab_size, int seq_len)
        : dim(dim), hidden_dim(hidden_dim), n_layers(n_layers),
          n_heads(n_heads), n_kv_heads(n_kv_heads), 
          vocab_size(vocab_size), seq_len(seq_len) {}
};

#endif
