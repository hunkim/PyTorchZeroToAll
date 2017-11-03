# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import gzip
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    # Initialize your data, download, etc.

    def __init__(self, filename="./data/shakespeare.txt.gz"):
        self.len = 0
        with gzip.open(filename, 'rt') as f:
            self.targetLines = [x.strip() for x in f if x.strip()]
            self.srcLines = [x.lower().replace(' ', '')
                             for x in self.targetLines]
            self.len = len(self.srcLines)

    def __getitem__(self, index):
        return self.srcLines[index], self.targetLines[index]

    def __len__(self):
        return self.len


# Test the loader
if __name__ == "__main__":
    dataset = TextDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=3,
                              shuffle=True,
                              num_workers=2)

    for i, (src, target) in enumerate(train_loader):
        print(i, "data", src)
