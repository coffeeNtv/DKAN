import os
from functools import partial
import timm
import torch
from models.resnet_custom import resnet50_baseline
import os

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = './home/ckpt/conch/pytorch_model.bin'
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
       # from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = '/home/ckpt/pytorch_model.bin'
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH


def get_encoder(model_name):
    print('loading model checkpoint')
    if model_name == 'res50':
        model = resnet50_baseline(pretrained=True)
        #model = model.to(device)
    elif model_name == 'plip':
        from .plip_wrapper import PLIPWrapper
        model = PLIPWrapper("vinid/plip")
    elif model_name == 'uni':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    return model


def get_backbone(model_name):
    print('loading model checkpoint')
    if model_name == 'res50':
        model = resnet50_baseline(pretrained=True)
    elif model_name == 'uni':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5,
                            img_size=224,
                            patch_size=16,
                            num_classes=0,
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cuda"), strict=True)
    elif model_name == 'conch':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    

    return model
