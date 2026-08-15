import torch

torch.manual_seed(0)

# frozen pretrained weight
d, k, r = 2, 3, 1
W0 = torch.tensor([[1., 2., 2.], [0., 3., 4.]])
V = W0.clone()  # direction, frozen

m = torch.linalg.norm(V, dim=0, keepdim=True).clone()  # magnitude, trainable
m.requires_grad = True

A = torch.randn(r, k) * 0.01
A.requires_grad = True
B = torch.zeros(d, r, requires_grad=True)

x = torch.tensor([1., 0.5, -1.])
target = torch.tensor([2.0, 1.0])

lr = 0.1
for step in range(500):
    delta_V = B @ A
    V_new = V + delta_V
    col_norm = torch.linalg.norm(V_new, dim=0, keepdim=True)  # per-column norm
    direction = V_new / col_norm
    W = m * direction
    output = W @ x

    loss = torch.mean((target - output) ** 2)
    loss.backward()

    with torch.no_grad():
        A -= lr * A.grad
        B -= lr * B.grad
        m -= lr * m.grad
    A.grad.zero_()
    B.grad.zero_()
    m.grad.zero_()

    if step % 100 == 0:
        print(step, loss.item())