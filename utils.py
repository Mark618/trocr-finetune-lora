def num_trainable_parameters(model,label):
    parameters,trainable = 0,0
    
    for _,p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0
        
    return f"{label} trainable parameters:{trainable:,}/{parameters:,} ({100*trainable/parameters:.2f}%)"
