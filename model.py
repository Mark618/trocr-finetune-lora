from peft import LoraConfig,get_peft_model


def model_create(opt,processor,model):   
        
    #set special tokens used for creating the decoder_input_ids from the labels
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

    # Encoder adapter config
    if opt['encoder_train'] == "dora":    
        en_config = LoraConfig(
            r=opt["encoder_rank"],
            lora_alpha=opt["encoder_alpha"],
            target_modules=opt["encoder_target_modules"],
            lora_dropout=0.1,
            bias="none",
            # modules_to_save=["pooler"],
            use_dora=True
        )
    else:
        en_config = LoraConfig(
            r=opt["encoder_rank"],
            lora_alpha=opt["encoder_alpha"],
            target_modules=opt["encoder_target_modules"],
            lora_dropout=0.1,
            bias="none",
            # modules_to_save=["pooler"],
            use_dora=False
        )

    # Decoder adapter config
    if opt['decoder_train'] == "dora":
        de_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            # modules_to_save=["pooler"],
            use_dora=True
        )
    else:
        de_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            # modules_to_save=["pooler"],
            use_dora=False
        )
        
        
    if opt['encoder_train'] != 'None':
        model.encoder= get_peft_model(model.encoder,en_config)
        
        if opt['encoder_train'] == 'lora-fa':
            for layer_name, p in model.encoder.named_parameters(): 
                if "lora_A" in layer_name:
                    p.requires_grad=False
    else:
        for layer_name, p in model.encoder.named_parameters():
            p.requires_grad=False
                    
        
    if opt['decoder_train'] != 'None':
        model.decoder= get_peft_model(model.decoder,de_config)
        
        if opt['decoder_train'] == 'lora-fa':
            for layer_name, p in model.decoder.named_parameters(): 
                if "lora_A" in layer_name:
                    p.requires_grad=False
    else:
        for layer_name, p in model.decoder.named_parameters():
            p.requires_grad=False
            
    return model