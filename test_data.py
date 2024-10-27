from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from ldm.util import instantiate_from_config

config_path = 'configs/spring_op/local_v15.yaml'
batch_size = 5
num_workers = 1
gpus = -1

config = OmegaConf.load(config_path)
dataset = instantiate_from_config(config["data"])
print(len(dataset))
dataloader = DataLoader(
    dataset,
    num_workers=num_workers,
    batch_size=batch_size,
    pin_memory=True,
    shuffle=True,
    persistent_workers=True,
)

for batch in dataloader:
    # `batch` is a list of dictionaries, each containing 'jpg', 'txt', 'local_conditions', and 'global_conditions'
    for sample in batch:
        # print(sample.keys())
        images = sample['jpg']  # Image data
        txt = sample['txt']  # Text/annotation data
        local_conditions = sample['local_conditions']  # Local conditions (optical flow and encoded frame)
        global_conditions = sample['global_conditions']  # Global conditions
        
        # Now you can use images, local_conditions, global_conditions in your model
        print("Images shape:", np.array(images).shape)
        print(txt)
        print("Local conditions shape:", np.array(local_conditions).shape)
        print("Global conditions shape:", np.array(global_conditions).shape)
