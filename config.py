from dataclasses import dataclass
@dataclass
class TinyMMT_JEPAConfig:
    d_model:      int   = 128
    n_heads:      int   = 4
    trunk_layers: int   = 3
    
    pred_layers:  int   = 4      # narrow predictor
    pred_dim:     int   = 128    # d_model // 2

    dec_layers:   int   = 3
    dec_dim:      int   = 128

    n_mels:       int   = 80
    n_langs:      int   = 2      # eng, twi
    n_mods:       int   = 2      # text, audio
    sample_rate:  int   = 16_000 # audio sample rate
		
    dropout:      float = 0.15
    ema_decay:    float = 0.996
    vocab_size:   int   = 16_000
    max_seq_len:  int   = 500   # upper bound for PE cache