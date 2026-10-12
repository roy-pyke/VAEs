from torchvision import datasets, transforms
from torch.utils.data import DataLoader

dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

image, label = dataset[0]

loader = DataLoader(
    dataset, 
    batch_size = 64,
    shuffle = True,
    num_workers = 0
)

x, labels = next(iter(loader))

print("datasets size: ", len(dataset))
print("image shape: ", image.shape)
print("image dtype: ", image.dtype)
print("pixel range: ", image.min().item(), image.max().item())
print("class: ", dataset.classes[label])

print("batch shape: ", x.shape)
print("label shape: ", labels.shape)    