import os
import re
import logging
import sys
import argparse
import pandas as pd
import numpy as np
from jiwer import cer,wer
import cv2
from tqdm import tqdm
from PIL import Image
import plotly.express as px
import torch
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from peft import PeftModel

class TestModel:
    def __init__(self,exp_name,trmodel):
        self.exp_name = exp_name
        self.trmodel = trmodel
        self.device = trmodel[1].device
        
    def postprocess_CA(self,text):
        text = text.lower()
        text = re.sub(r'[ $/x-zhk=*#sa-z-]',"",text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def postprocess_date(self,text):
        text = text.lower()
        text = re.sub(r"[.,*#]", "", text)
        text = re.sub(r"[/-]", " ", text)
        text = re.sub(r"[a-zA-Z]", "", text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def postprocess_LA(self,text):
        text = text.lower()
        text = re.sub(r"hk", "", text)
        text = re.sub(r'\*', "", text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def postprocess_payee(self,text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def cal_cer(self,label,ocr_result):
        if pd.isna(ocr_result) or pd.isnull(ocr_result):
            return np.nan       
        
        return round(cer(str(label).lower(), str(ocr_result).lower()),2)

    def calculate_wer(self,label, ocr_result):
        if pd.isna(ocr_result) or pd.isnull(ocr_result):
            return np.nan
        return round(wer(str(label).lower(), str(ocr_result).lower()),2)

    def cal_cer_wer_trocr(self,temp_df):          
        temp_df[f'{self.exp_name}_cer'] = temp_df.apply(lambda row: self.cal_cer(row['labels'], row[self.exp_name]), axis=1) #Value
        temp_df[f'{self.exp_name}_wer'] = temp_df.apply(lambda row: self.calculate_wer(row['labels'], row[self.exp_name]), axis=1)        
        return temp_df
    

    def test_func(self,input_arr):
        df = pd.read_csv(os.path.join(input_arr[0],input_arr[1])) 
        df = df[df['file_pth'].isin(["CA","date"])]
        base_dir = input_arr[1]
        ocr_result = []
                    
        for i, rows in tqdm(df.iterrows(),total=df.shape[0],desc=f"Processing {base_dir}"):
            f_name = rows['file_name']     
                
            ori_img= cv2.imread(os.path.join(input_arr[0],rows['file_pth'],f_name))          
        
            if ori_img is not None:
                pixel_values = self.trmodel[0](images=ori_img, return_tensors="pt").pixel_values.to(self.device)
                
                generated_ids = self.trmodel[1].generate(pixel_values,num_beams=4)
                generated_text = self.trmodel[0].batch_decode(generated_ids, skip_special_tokens=True)[0]              

                ocr_result.append(generated_text)
            else:
                ocr_result.append(None)                
           
        df[f"{self.exp_name}"] = ocr_result
        
            
        return df 
    
    def legal_twice(self,input_arr):
        df = pd.read_csv(os.path.join(input_arr[0],input_arr[1])) 
       
        filter_df = df.loc[df['file_pth']=='legal_amount']
        base_dir = input_arr[1].split('.')[0]
        category ='bkp_legal'
        final_output = []
        combined_images = {}                
        for filename in tqdm(sorted(os.listdir(os.path.join(input_arr[0],base_dir,category))),desc="Processing legal amount twice"):
            prefix = filename.split('.')[0]
            
            img = cv2.imread(os.path.join(input_arr[0],base_dir,category,filename))
            if img is not None:
                pixel_values = self.trmodel[0](images=img, return_tensors="pt").pixel_values.to(self.device)
                generated_ids = self.trmodel[1].generate(pixel_values,num_beams=4)
                
                generated_text = self.trmodel[0].batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
            if prefix in combined_images:
                combined_images[prefix].append(generated_text)
            else:                
                combined_images[prefix]=[generated_text]
                
        for prefix,txt in combined_images.items():
            combined_text = " ".join(txt)
            final_output.append([prefix,combined_text])
            
        temp_df = pd.DataFrame(final_output,columns=['file_name',f"{self.exp_name}"])   
        filter_df.loc[:,"basename"] = filter_df["file_name"].str.split(".").str[0]
        filter_df = pd.merge(filter_df, temp_df, left_on="basename", right_on="file_name", how="left")
        filter_df = filter_df.drop(columns=["basename", "file_name_y"]).rename(columns={"file_name_x": "file_name"})
        
        return filter_df        
      
                
    
    def post_proc(self,result_df):
        # CA
        result_df.loc[result_df['file_pth']=='courtesy_amount','labels'] = result_df.loc[result_df['file_pth']=='courtesy_amount','labels'].apply(self.postprocess_CA)
        result_df.loc[result_df['file_pth']=='courtesy_amount',self.exp_name] = result_df.loc[result_df['file_pth']=='courtesy_amount',self.exp_name].apply(self.postprocess_CA)

        # Date
        result_df.loc[result_df['file_pth']=='date','labels'] = result_df.loc[result_df['file_pth']=='date','labels'].apply(self.postprocess_date)
        result_df.loc[result_df['file_pth']=='date',self.exp_name] = result_df.loc[result_df['file_pth']=='date',self.exp_name].apply(self.postprocess_date)

        #Legal Amount
        result_df.loc[result_df['file_pth']=='legal_amount','labels'] = result_df.loc[result_df['file_pth']=='legal_amount','labels'].apply(self.postprocess_LA)
        result_df.loc[result_df['file_pth']=='legal_amount',self.exp_name] = result_df.loc[result_df['file_pth']=='legal_amount',self.exp_name].apply(self.postprocess_LA)

        #Payee
        result_df.loc[result_df['file_pth']=='payee','labels'] = result_df.loc[result_df['file_pth']=='payee','labels'].apply(self.postprocess_payee)
        result_df.loc[result_df['file_pth']=='payee',self.exp_name] = result_df.loc[result_df['file_pth']=='payee',self.exp_name].apply(self.postprocess_payee)
        
        return result_df

    def check_func (self,input_arr):      
        test_all=self.test_func(input_arr)  
        return test_all
    
class RIHKV2:
    def __init__(self,exp_name,trmodel):
        self.exp_name = exp_name
        self.trmodel = trmodel
        self.device = trmodel[1].device
        
    def test_func(self,input_df):
        df = input_df.loc[input_df['file_pth'] !='legal_amount']        
        ocr_result = []
                    
        for i, rows in tqdm(df.iterrows(),total=df.shape[0],desc=f"Processing RIHKV2"):               
        
            ori_img= Image.open(rows['abs_path'])          
        
            if ori_img is not None:
                pixel_values = self.trmodel[0](images=ori_img, return_tensors="pt").pixel_values.to(self.device)
                
                generated_ids = self.trmodel[1].generate(pixel_values,num_beams=4)
               
                generated_text = self.trmodel[0].batch_decode(generated_ids, skip_special_tokens=True)[0]               
                ocr_result.append(generated_text)                
           
        df[f"{self.exp_name}"] = ocr_result
        
            
        return df 
    
    def legal_twice(self,input_df,base_dir):        
       
        filter_df = input_df.loc[input_df['file_pth']=='legal_amount']
        category ='legal_amount'
        final_output = []        
        for folder_num in range(1,14):  
            combined_images = {}              
            for filename in tqdm(sorted(os.listdir(os.path.join(base_dir,str(folder_num),category))),desc="Processing legal amount twice"):
                prefix = filename.split('.')[0]
                
                img = Image.open(os.path.join(base_dir,str(folder_num),category,filename))
                if img is not None:
                    
                    pixel_values = self.trmodel[0](images=img, return_tensors="pt").pixel_values.to(self.device)
                    generated_ids = self.trmodel[1].generate(pixel_values,num_beams=4)
             
                    generated_text = self.trmodel[0].batch_decode(generated_ids, skip_special_tokens=True)[0]                   
                    
                        
                if prefix in combined_images:
                    combined_images[prefix].append(generated_text)
                else:                
                    combined_images[prefix]=[generated_text]
                    
            for prefix,txt in combined_images.items():
                combined_text = " ".join(txt)
                final_output.append([folder_num,prefix,combined_text])
            
        temp_df = pd.DataFrame(final_output,columns=['set','file_name',f"{self.exp_name}"])   
        # filter_df[self.exp_name] = final_output
        filter_df.loc[:,"basename"] = filter_df["file_name"].str.split(".").str[0]
        filter_df = pd.merge(filter_df, temp_df, left_on=["set","basename"], right_on=["set","file_name"], how="left")
        filter_df = filter_df.drop(columns=["basename", "file_name_y"]).rename(columns={"file_name_x": "file_name"})
        
        return filter_df
    
    def check_func (self,input_arr):      
        test_all=self.test_func(input_arr) 
       
        
        legal_seperate = self.legal_twice(input_arr)        
        conc_df = pd.concat([test_all,legal_seperate],ignore_index=True)
         
        # return test_all,legal_seperate
        return conc_df

CURRENT_DIR = os.curdir

if __name__ == '__main__':
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
    
    # Parser
    parser = argparse.ArgumentParser(description="Testing model performance")
    
    parser.add_argument('--model', '-m', help="Model name")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(os.path.join(CURRENT_DIR,'saved_models',"trocr-large-handwritten_half_prec"),torch_dtype=torch.float16)
    
    processor = TrOCRProcessor.from_pretrained(os.path.join(CURRENT_DIR,'saved_models',"trocr-large-handwritten"))
    
    model_name = args.model
    if os.path.exists(os.path.join(CURRENT_DIR,'saved_models',model_name,"decoder")):
        logger.info("Loading encoder and decoder adapter")
        model.encoder.load_adapter(os.path.join(CURRENT_DIR,'saved_models',model_name,"encoder"))
        model.decoder.load_adapter(os.path.join(CURRENT_DIR,'saved_models',model_name,"decoder"))
    else:
        logger.info("Loading encoder adapter")
        test_model = PeftModel.from_pretrained(model,os.path.join(CURRENT_DIR,'saved_models',model_name))
    
  # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    model.to(device)
    
    test_model= TestModel(model_name,[processor,model])
    temp_df = pd.read_csv( os.path.join(CURRENT_DIR,"rihkv2","rihkv2.csv"))
    
    target_pth = os.path.join(CURRENT_DIR,"rihkv2")
    
    logger.info("Start testing on RIHKv2")
    test= RIHKV2(model_name,[processor,model])
    rihkv2_output = test.test_func(input_df=temp_df)
    rihkv2_output_legal = test.legal_twice(temp_df,target_pth)
    
    conc_df = pd.concat([rihkv2_output,rihkv2_output_legal],ignore_index=True)
    temp_conc_df = test_model.post_proc(conc_df)
    temp_conc_df=test_model.cal_cer_wer_trocr(temp_conc_df)
    temp_conc_df = temp_conc_df.drop('abs_path',axis=1)
    
    logger.info("Saving result")
    temp_conc_df.to_csv(os.path.join(CURRENT_DIR,'saved_models',model_name,"rihkv2.csv"),index=False)
    temp_conc_df['flag'] = (temp_conc_df[f'{model_name}_cer'] > 0).astype(int)
    fig=px.bar(temp_conc_df.groupby(['file_pth','flag'], as_index=False).size(),x="file_pth",y="size",color="flag",text="size",labels={'file_pth': 'Category', 'size': 'Count', 'flag': 'Flag'},title=model_name)
    fig.write_image(os.path.join(CURRENT_DIR,'saved_models',model_name,"rihkv2.png"))
    logger.info("Done")