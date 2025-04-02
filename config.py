# evolingua/config.py
class EvoLinguaConfig:
    """Configuration for the EvoLingua model."""
    vocab_size = 100000  # Adjust based on tokenizer
    embed_dim = 4096     # Hidden size
    num_heads = 32       # Attention heads
    num_layers = 80      # Transformer layers
    num_experts = 128    # Total MoE experts
    experts_per_token = 2  # Activated experts per token
    kv_compress_dim = 256  # KV compression dimension
    mtp_depth = 4        # Multi-Token Prediction depth
    max_seq_len = 128000 # Maximum sequence length
    dropout = 0.1        # Dropout rate
    bias_update_speed = 0.01  # Load balancing adjustment speed
