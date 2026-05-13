import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Dataset
class TextDataset(Dataset):
    def __init__(self,root_dir,df,processor,img_transform=False,max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.img_trans = img_transform
        self.max_target_length = max_target_length
        self.img_trans_func = transforms.Compose([
                                transforms.RandomRotation((5,5)),
                                transforms.RandomAdjustSharpness(1.5),
                                transforms.RandomAutocontrast(0.6),
                                transforms.GaussianBlur(kernel_size=(5,5))
                            ])      
        
    def __len__(self):
        # return 8
        return len(self.df)
    
    def __getitem__(self,idx):
        # Get file name and text
        file_name = self.df['file_name'][idx]        
        text = self.df['labels'][idx]
        image = Image.open(os.path.join(self.root_dir,file_name)).convert('RGB')
        if self.img_trans:
            image = self.img_trans_func(image)
            
        pixel_values = self.processor(image,return_tensors="pt").pixel_values
        
        # add labels
        labels = self.processor.tokenizer(str(text),padding="max_length",max_length = self.max_target_length).input_ids
        
        labels = [label if label !=self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
    

