import torch

def run(A, B, C, **kwargs):
    # C = A @ B.T, A:[M,K] B:[N,K] C:[M,N] fp16
    torch.matmul(A, B.T, out=C)
