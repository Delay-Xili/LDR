import numpy as onp
import torch

N = 64
D = 128

I = torch.eye(D).cuda()
V = torch.randn(D, N).cuda()
alpha = 0.1


# def one(V):
#     A = I + alpha * V @ V.T
#     return 2 * torch.trace(torch.log(torch.cholesky(A)))


def two(V):
    A = I + alpha * V @ V.T
    return torch.logdet(A)


def three(V):
    A = I + alpha * V @ V.T
    return 2 * torch.sum(
        torch.log(
            torch.diag(
                torch.cholesky(
                    A,
                    # upper=True,
                )
            )
        )
    )


import timeit

for f in [two, three]:
    f(V)
    print(timeit.timeit(lambda: f(V), number=1000))