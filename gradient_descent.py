import torch
### Gradient descent implemented in pytorch ### 
# let's say we want to find the minimum value for 
# function: f(x) = x^4 - 5x^3 + 3x^2 + -x + 3

def f(x):
    return 1*(x**4) - 5*(x**3) + 3*(x**2) - x + 3

# first initialize a tensor from an initial guess value
initial_guess = 2.0
x0 = torch.tensor(initial_guess)
x0.requires_grad = True

# next, set hyperparams
n_iterations = 25
learning_rate = 0.01
optimizer = torch.optim.SGD([x0], lr=learning_rate)
for i in range(n_iterations):
    loss = f(x0)
    
    # compute the gradient (by pytorch), and do gradient descent to update x0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # should see the loss decreases
    print("iteration:%04d x=%.4f f(x)=%.4f"%(i, x0.item(), loss.item()))

print("The minimum val is achieved at x=%.3f, with minimum f(x)=%.3f"%(x0.item(), f(x0).item()))