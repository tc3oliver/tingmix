import torch
from embed2asr.reader import Reader

from tingmix.models.large.attention import AttDecoder, reader_states


def test_decoder_padding_and_reader_hook():
    torch.manual_seed(0)
    dec = AttDecoder(16, 10, d=32, layers=2, heads=4, ff=64, dropout=0.0).eval()
    mem = torch.randn(2, 7, 16)
    ce, _ = dec.loss(mem, torch.tensor([7, 5]), [[1, 2, 3], [4]])
    assert torch.isfinite(ce)
    # padding (memory and hypotheses) must not change a score
    a = dec.score(mem[1], 5, [[4], [1, 2, 3]])
    b = dec.score(mem[1, :5], 5, [[4]])
    assert abs(a[0] - b[0]) < 1e-4
    # the hook only observes: CTC logits are unchanged
    r = Reader(16, 10, d=8, gru=4, blocks=1).eval()
    lg, h = reader_states(r, mem, torch.tensor([7, 5]))
    assert h.shape[-1] == 8 and torch.allclose(lg, r(mem, torch.tensor([7, 5])))
