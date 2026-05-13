import os
import shutil
import logging
import sys
import yaml
from test import TestModel,RIHKV2
from utils import num_trainable_parameters
from model import model_create
from dataset import TextDataset

from PIL import Image
import pandas as pd
from tqdm import tqdm
from jiwer import cer,wer

import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

import transformers
from transformers import TrOCRProcessor,VisionEncoderDecoderModel
from peft import LoraConfig,get_peft_model,inject_adapter_in_model
from evaluate import load

# Set logging level
transformers.logging.set_verbosity_error()

def get_config(file_path):
    try:
        with open(file_path,'r', encoding="utf8") as stream:
            opt = yaml.safe_load(stream)
        return opt
    except FileNotFoundError:
        logger.error(f"Config file not found: {file_path}")
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while reading config file {file_path}: {e}")
        
def check_folders_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File/Folder not found: {path}")
    logger.info("All required file/folder found")

CURRENT_DIR = os.curdir

cer_metric = load("cer")
def compute_cer(pred_ids, label_ids):
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return cer


def train_val(opt,train_loader,val_loader,train_model):  
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=opt["lr"] ,weight_decay=0.0005)  
    result = []
    try:
        for epoch in range(opt["num_epochs"]):  # loop over the dataset multiple times
    # train
            print("Start training")
            train_model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader):
                # get the inputs
                for k,v in batch.items():
                    batch[k] = v.to(device)

                # forward + backward + optimize
                outputs = train_model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()

            print(f"Loss epoch {epoch}:", train_loss/len(train_dataloader))
        
            # evaluate
            print("Start validation")
            train_model.eval()
            valid_cer = 0.0
            val_loss = 0.0
            
            with torch.no_grad():        
                for batch in tqdm(val_loader):
                    for k,v in batch.items():
                        batch[k] = v.to(device)                 
                    outputs_l = train_model(**batch)               
                    loss = outputs_l.loss            
                    val_loss += loss.item()                    
                    
            print(f"Validation loss epoch {epoch}: {val_loss/ len(val_loader)}. CER: {valid_cer/len(val_loader)}")           
            
            #save training log
            result.append([epoch,train_loss/len(train_loader),val_loss/len(val_loader),valid_cer/len(val_loader)])
        
        # if opt['encoder_train'] != "None":
            # train_model.encoder.save_pretrained(os.path.join(MODEL_SAVED_PATH,f"{opt["experiment_name"]}","encoder"),save_embedding_layers=False)
            
        if opt['decoder_train'] != "None":
            train_model.decoder.save_pretrained(os.path.join(MODEL_SAVED_PATH,f"{opt["experiment_name"]}","decoder"),save_embedding_layers=False)
            train_model.encoder.save_pretrained(os.path.join(MODEL_SAVED_PATH,f"{opt["experiment_name"]}","encoder"),save_embedding_layers=False)
        else:
            train_model.save_pretrained(os.path.join(MODEL_SAVED_PATH,f"{opt["experiment_name"]}"))
            
        logger.info(f"Model saved in {os.path.join(MODEL_SAVED_PATH,f"{opt["experiment_name"]}")}")
        df = pd.DataFrame(result,columns=["epoch",'train_loss',"val_loss","CER"])
        df.to_csv(os.path.join(MODEL_SAVED_PATH,opt["experiment_name"],f"{opt["experiment_name"]}.csv"))       
        
    except KeyboardInterrupt:
        if opt['decoder_train'] != "None":
            train_model.decoder.save_pretrained(os.path.join(MODEL_SAVED_PATH,f"{opt["experiment_name"]}","decoder"),save_embedding_layers=False)
            train_model.encoder.save_pretrained(os.path.join(MODEL_SAVED_PATH,f"{opt["experiment_name"]}","encoder"),save_embedding_layers=False)
        else:
            train_model.save_pretrained(os.path.join(MODEL_SAVED_PATH,f"{opt["experiment_name"]}"))
        logger.warning(f"Model saved in {os.path.join(MODEL_SAVED_PATH,f"{opt["experiment_name"]}")}. Task Failed")
        df = pd.DataFrame(result,columns=["epoch",'train_loss',"val_loss","CER"])
        df.to_csv(os.path.join(MODEL_SAVED_PATH,opt["experiment_name"],f"{opt["experiment_name"]}.csv"))

    return train_model

if __name__ == "__main__":
    # Setup logging
    logger = logging.getLogger('my_app')
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler('app.log','w')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Read config file
    logger.info('Setting up')
    logger.info('Load config file')    
    
    opt = get_config(os.path.join(CURRENT_DIR,"config","config.yaml"))
    if opt:
        logger.info("Config loaded successfully")
    else:
        logger.error("Failed to load config")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TRAIN_PATH = os.path.join(CURRENT_DIR,opt['train_data'],"img")
    VAL_PATH = os.path.join(CURRENT_DIR,opt['valid_data'],"img")
    TRAIN_CSV_PATH = os.path.join(CURRENT_DIR,opt['train_data'],"labels.csv")
    VAL_CSV_PATH = os.path.join(CURRENT_DIR,opt['valid_data'],"labels.csv")
    MODEL_SAVED_PATH = os.path.join(CURRENT_DIR,"saved_models")

    try:
        im_path = [TRAIN_PATH,VAL_PATH,TRAIN_CSV_PATH,VAL_CSV_PATH]
        check_folders_exist(im_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        exit()
   
    # Read csv file
    logger.info("Reading training labels csv")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    logger.info("Reading training labels csv DONE")
    logger.info("Reading validation labels csv")
    val_df = pd.read_csv(VAL_CSV_PATH)
    logger.info("Reading validation labels csv DONE")
    

    processor = TrOCRProcessor.from_pretrained(os.path.join(MODEL_SAVED_PATH,"trocr-large-handwritten"))
    logger.info("Create Dataset and DataLoader")
    train_dataset = TextDataset(root_dir=TRAIN_PATH,df=train_df,processor=processor,img_transform=opt["img_augmentation"])
    val_dataset = TextDataset(root_dir=VAL_PATH,df=val_df,processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt["batch_size"])
    logger.info("Create Dataset and DataLoader DONE")
    
    logger.info("Loading Base Model")
    if opt["half_prec"]:
        model = VisionEncoderDecoderModel.from_pretrained(os.path.join(MODEL_SAVED_PATH,"trocr-large-handwritten_half_prec"),torch_dtype=torch.float16)
    else:
        model = VisionEncoderDecoderModel.from_pretrained(os.path.join(MODEL_SAVED_PATH,"trocr-large-handwritten"))
    logger.info("Loading Base Model DONE")    
    logger.info("Model total trainable parameters before Peft:")
    logger.info(num_trainable_parameters(model,"Whole model"))
    logger.info(num_trainable_parameters(model.encoder,"Encoder"))
    logger.info(num_trainable_parameters(model.decoder,"Decoder"))
    
    logger.info("Loading Peft")
    model = model_create(opt,processor,model)
                    
    model.to(device)
    logger.info("Loading Peft DONE")
    logger.info("Model total trainable parameters after Peft:")
    logger.info(num_trainable_parameters(model,"Whole model"))
    logger.info(num_trainable_parameters(model.encoder,"Encoder"))
    logger.info(num_trainable_parameters(model.decoder,"Decoder"))
    
    logger.info("Start Training")
    trained_model = train_val(opt,train_dataloader,val_dataloader,model)
    logger.info("Training DONE")
    
    shutil.move("app.log",os.path.join(MODEL_SAVED_PATH,"logfile.log"))