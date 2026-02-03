import torch

from my_dlinear import DLinear
from my_dlinear.utils import choose_dlinear_kernel


def main():
    B = 4
    seq_len = 12
    pred_len = 5
    enc_in = 3

    k = choose_dlinear_kernel(seq_len, base=25)
    model = DLinear(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, kernel_size=k, individual=True)

    x = torch.randn(B, seq_len, enc_in)
    y = model(x)

    print("x:", x.shape)
    print("y:", y.shape)


if __name__ == "__main__":
    main()
