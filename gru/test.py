from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()
c = Tensor.zeros(3,3)
c = c + x
c = c + z
c.sum().backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
print(c.grad)
