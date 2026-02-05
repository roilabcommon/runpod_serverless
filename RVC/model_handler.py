import torch
import numpy as np
import huggingface_hub
import zipfile
import os
from collections import OrderedDict

def model_info(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    info = {
        'config': model['config'],
        'info': model['info'],
        'epochs': model['info'].split('epoch')[0],
        'sr': model['sr'],
        'f0': model['f0'],
        'size': model['size'] if 'size' in model['weight'] else 'fp32',
    }
    return info

def merge(path1, path2, alpha1, sr, f0, info, name, version):
    try:
        def extract(ckpt):
            a = ckpt["model"]
            opt = OrderedDict()
            opt["weight"] = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = a[key]
            return opt

        ckpt1 = torch.load(path1, map_location="cpu")
        ckpt2 = torch.load(path2, map_location="cpu")
        cfg = ckpt1["config"]
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            # try:
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key][:min_shape0].float())
                    + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        # except:
        #     pdb.set_trace()
        opt["config"] = cfg
        """
        if(sr=="40k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 40000]
        elif(sr=="48k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10,6,2,2,2], 512, [16, 16, 4, 4], 109, 256, 48000]
        elif(sr=="32k"):opt["config"] = [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 4, 2, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 32000]
        """
        opt["sr"] = sr
        opt["f0"] = 1 if f0 == "Yes" else 0
        opt["version"] = version
        opt["info"] = info
        torch.save(opt, "models/" + name + ".pth")
        return "models/" + name + ".pth"
    except:
        return "Fail to merge the models. The model architectures are not the same." # <- L if u see this u suck

def model_quant(model_path, size):
    """
    Quantize the model to a lower precision. - this is the floating point version

    Args:
        model_path: str, path to the model file
        size: str, one of ["fp2", "fp4", "fp8", "fp16"]
    
    Returns:
        str, message indicating the success of the operation
    """
    size_options = ["fp2", "fp4", "fp8", "fp16"]
    if size not in size_options:
        raise ValueError(f"Size must be one of {size_options}")
    
    model_base = torch.load(model_path, map_location=torch.device('cpu'))
    model = model_base['weight']
    #model = json.loads(json.dumps(model))

    if size == "fp16":
        for key in model.keys():
            model[key] = model[key].half() # 16-bit floating point
    elif size == "fp8":
        for key in model.keys():
            model[key] = model[key].half().half() # 8-bit floating point <- this is the most common one
    elif size == "fp4":
        for key in model.keys():
            model[key] = model[key].half().half().half() # 4-bit floating point <- ok maybe you're mentally ill if you choose this (very low precision)
    elif size == "fp2":
        for key in model.keys():
            model[key] = model[key].half().half().half().half() # 2-bit floating point <- if you choose this you're a fucking dickhead coming
    
    print(model_path)
    output_path = model_path.split('.pth')[0] + f'_{size}.pth'
    output_style = {
        'weight': model,
        'config': model_base['config'],
        'info': model_base['info'],
        'sr': model_base['sr'],
        'f0': model_base['f0'],
        'credits': f"Quantized to {size} precision, using Ilaria RVC, (Mikus's script)",
        "size": size
    }
    torch.save(output_style, output_path)

    #AmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithraxAmerithrax
    # our data isnt safe anymore currently typing this and there is a 100% chance that it'll be stolen and used for training another fucking dogshit language model by a horrible company like openai
    # i say this as a person who communicates with microsoft and i will stop mentioning this as they're so closely tied together nowadays
    # as fred durst has said - "That's your best friend and your worst enemy - your own brain." - keep your shit local and never trust scumbag companies even if they make the models oss - they're stealing data
    # this is probably the only rant i'll have in this entire space and i put it in a notable spot

    return "Model quantized successfully" # <- enjoy this fucking hot shit that looks like a steaming turd paired with skibidi toilet and the unibomber

def upload_model(repo, pth, index, token):
    """
    Upload a model to the Hugging Face Hub

    Args:
        repo: str, the name of the repository
        pth: str, path to the model file
        index: str, the index of the model in the repository
        token: str, the API token
    
    Returns:
        str, message indicating the success of the operation
    """
    readme = f"""
    # {repo}
    This is a model uploaded by Ilaria RVC, using Mikus's script.
    """
    repo_name = repo.split('/')[1]
    with zipfile.ZipFile(f'{repo_name}.zip', 'w') as zipf:
        zipf.write(pth, os.path.basename(pth))
        zipf.write(index, os.path.basename(index))
        zipf.writestr('README.md', readme)
    
    huggingface_hub.HfApi().create_repo(token=token, name=repo, exist_ok=True)
    huggingface_hub.HfApi().upload_file(token=token, path=f'{repo.split("/")[1]}.zip', repo_id=repo)
    os.remove(f'{repo.split("/")[1]}.zip')
    return "Model uploaded successfully"