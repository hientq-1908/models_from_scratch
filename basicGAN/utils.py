import torch

def save_checkpoint(model, optimizer, path):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state_dict, path)
    print('saved successfully!')

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer