from pathlib import Path

import sentencepiece as spm


SPECIAL_TOKEN_IDS = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<pad>": 3,
    "[CLS]": 4,
    "[SEP]": 5,
    "[MASK]": 6,
}


class SpliNetTokenizer:
    """Small direct wrapper around the released SentencePiece model."""

    def __init__(self, model_file):
        self.model_file = str(Path(model_file))
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)
        if self.sp.get_piece_size() != 32000:
            raise ValueError("Expected a 32,000-piece SentencePiece vocabulary")
        for token, expected in SPECIAL_TOKEN_IDS.items():
            actual = int(self.sp.piece_to_id(token))
            if actual != expected:
                raise ValueError(
                    f"Special-token mismatch: {token} -> {actual}, expected {expected}"
                )

    def encode(self, text, add_special_tokens=True):
        ids = self.sp.encode(" ".join((text or "").lower().split()), out_type=int)
        if add_special_tokens:
            return [SPECIAL_TOKEN_IDS["[CLS]"], *ids, SPECIAL_TOKEN_IDS["[SEP]"]]
        return ids

    def decode(self, ids):
        return self.sp.decode(list(map(int, ids)))
