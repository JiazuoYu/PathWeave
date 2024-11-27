import os
import torch
import json

if __name__ == "__main__":
    ckpt_path = ''
    save_path = ''
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
        for name in checkpoint.keys():
            if 'adapter' in name or 'in_adapter' in name:
                data =  checkpoint[name]
                print("{}:{}".format(name, data))
        # data_dict = list(checkpoint.keys())
        # print(data_dict)
        # with open(save_path, 'w') as f:
        #     json.dump(data_dict, f, indent=4)
            # for k in checkpoint.keys():
                