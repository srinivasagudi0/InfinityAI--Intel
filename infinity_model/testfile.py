import torch

# i want to test and learn how to use torch

x = torch.tensor([1, 2, 3])
print(x)
print(x.shape)
print(x+10)


# now I will test nn.module

def learn_by_testing():
    # create a simple linear model
    model = torch.nn.Linear(1, 1) # input size 1, output size 1

    # create some dummy data
    x = torch.tensor([[1.0], [2.0])