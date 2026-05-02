from dataclasses import dataclass
@dataclass
class TinyMMT_JEPAConfig:
    d_model:      int   = 128
    n_heads:      int   = 4
    trunk_layers: int   = 3
    
    pred_layers:  int   = 4      # narrow predictor
    pred_dim:     int   = 128    # d_model // 2

    dec_layers:   int   = 5
    dec_dim:      int   = 128

    n_mels:       int   = 80
    n_langs:      int   = 2      # eng, twi
    n_mods:       int   = 2      # text, audio
    sample_rate:  int   = 16_000 # audio sample rate

    dropout:       float = 0.15
    sigreg_lambda: float = 0.02  # weight on SIGReg; pred weight is (1 - λ)
    sigreg_knots:  int   = 17
    vocab_size:   int   = 16_000
    max_seq_len:  int   = 500   # upper bound for PE cache


@dataclass
class ModelConfig:
    d_model:      int   = 512
    n_heads:      int   = 8
    trunk_layers: int   = 6
    
    pred_layers:  int   = 6      # narrow predictor
    pred_dim:     int   = 512    # d_model // 2

    dec_layers:   int   = 6
    dec_dim:      int   = 512

    vocab_size:   int   = 16_000
    n_mels:       int   = 80
    n_langs:      int   = 2      # eng, twi
    n_mods:       int   = 2      # text, audio

    dropout:       float = 0.15
    sigreg_lambda: float = 0.02  # weight on SIGReg; pred weight is (1 - λ)
    sigreg_knots:  int   = 17
    
    max_seq_len:  int   = 1500   # upper bound for PE cache
    sample_rate:  int   = 16_000 # audio sample rate