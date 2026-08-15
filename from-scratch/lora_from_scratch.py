import torch

torch.manual_seed(0)

# frozen pretrained weight
d, k, r = 2, 3, 1
W0 = torch.tensor([[1., 2., 2.], [0., 3., 4.]])

# lora matrices
A = torch.randn(r, k) * 0.01   # random init
A.requires_grad = True
B = torch.zeros(d, r, requires_grad=True)  # zero init -> update starts at 0

x = torch.tensor([1., 0.5, -1.])
target = torch.tensor([2.0, 1.0])  # made-up target output

lr = 0.1
for step in range(500):
    delta_W = B @ A
    output = (W0 + delta_W) @ x

    loss = torch.mean((target - output) ** 2)
    loss.backward()

    with torch.no_grad():
        A -= lr * A.grad
        B -= lr * B.grad
    A.grad.zero_()
    B.grad.zero_()

    if step % 100 == 0:
        print(step, loss.item())