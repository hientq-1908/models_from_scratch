import torch

def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f'save checkpoint f{filename}')