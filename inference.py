import torch
from decoder import Decoder, build_io_ids
from model import MMT_JEPA
from config import ModelConfig
import sentencepiece as spm


# Checkpoints
mmt_jepa_ckpt = "checkpoints/epoch010.pt"
decoder_ckpt = "checkpoints/jepa_ep003_decoder_epoch010.pt"
tokenizer_path = "tokenizer/tokenizer.model"

sp = spm.SentencePieceProcessor()
sp.Load(tokenizer_path)

cfg = ModelConfig()
jepa = MMT_JEPA(cfg)
jepa.load_state_dict(torch.load(mmt_jepa_ckpt))

model = Decoder(cfg, jepa)
model.load_state_dict(torch.load(decoder_ckpt, map_location="cpu"), strict=False)

del jepa
model.eval()


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device)

# Inputs
source = "The president of Ghana is a very strong man."
(
    src_lang,
    tgt_lang,
    src_mod,
    tgt_mod
) = build_io_ids(
    batch_size=1,
    device=device,
    src_lang="eng",
    tgt_lang="twi",
    src_mod="text",
    tgt_mod="text",
)

text_ids = sp.Encode(source)
text_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
print(text_ids.shape)
# start with <s>
bos_id = sp.bos_id()
tgt_text_ids = torch.tensor([bos_id], dtype=torch.long, device=device).unsqueeze(0)
for i in range(10):
    text_pred = model.forward(
        src_lang=src_lang,
        src_mod=src_mod,
        tgt_lang=tgt_lang,
        tgt_mod=tgt_mod,
        text_ids=text_ids,
        tgt_text_ids=tgt_text_ids,
    )
    next_token = text_pred.argmax(dim=-1)[:, -1]
    tgt_text_ids = torch.cat([tgt_text_ids, next_token.unsqueeze(0)], dim=1)
    print(sp.Decode(tgt_text_ids.squeeze(0).tolist()))