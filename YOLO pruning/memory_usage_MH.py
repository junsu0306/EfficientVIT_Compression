import copy

import torch
import torch.nn as nn

from model_compression.funcs4 import *


def measure_memory(x, layers, device):
    if isinstance(x, (tuple, list)):
        x = [item.cpu() for item in x]
    else:
        x = x.cpu()
    
    for layer in layers:
        layer.cpu()
    torch.cuda.empty_cache()
    
    before_memory = torch.cuda.memory_allocated(device)/1024**2
    #print("before_memory:", before_memory)
    
    with torch.no_grad():
        if isinstance(x, (tuple, list)):
            x = [item.to(device) for item in x]
        else:
            x = x.to(device)
            
        for layer in layers:
            layer.to(device)
            x = layer(x)

    after_memory = torch.cuda.memory_allocated(device)/1024**2
    
    if isinstance(x, (tuple, list)):
        x = (item.cpu() for item in x)
    else:
        x = x.cpu()
        
    for layer in layers:
        layer.cpu()
    torch.cuda.empty_cache()
    
    return x, after_memory - before_memory # forward 위해 x return




def extract_layers(model):
    layers_list = []
    
    for name, layer in model.named_children():  
        if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
            for sub_layer in layer:
                layers_list.append(sub_layer)
                
        else:
            layers_list.append(layer)
    
    return layers_list



# def measure_layer_memory(x, model, device):

#     torch.cuda.empty_cache()

#     layers_list = extract_layers(model.model) 

#     mem_list = []

#     layer_outputs = []

#     for i, layer in enumerate(layers_list):
#         if type(layer).__name__ == "Concat":
#             #print(f"Skipping layer {i} (concat)")
        
#             if i == 11:
#                 concat_inputs = [layer_outputs[6], layer_outputs[10]]
#             elif i == 14:
#                 concat_inputs = [layer_outputs[4], layer_outputs[13]]
#             elif i == 17:
#                 concat_inputs = [layer_outputs[12], layer_outputs[16]]
#             elif i == 20:
#                 concat_inputs = [layer_outputs[9], layer_outputs[19]]

#             x = layer(concat_inputs)
#             mem_list.append(0)
#             layer_outputs.append(x)
        
#             continue
            
#         if type(layer).__name__ == "Detect":
#             inputs = [layer_outputs[15], layer_outputs[18], layer_outputs[21]]

#             x, used_memory = measure_memory(inputs, [layer], device)
#             mem_list.append(used_memory)

#             #print(f" Layer {i} ({type(layer).__name__}) memory usage: {used_memory:.5f} MB")

#             continue
  
#         else:
#             x, used_memory = measure_memory(x, [layer], device)
#             mem_list.append(used_memory)
#             layer_outputs.append(x)
        
#             #print(f" Layer {i} ({type(layer).__name__}) memory usage: {used_memory:.5f} MB")

#     #print(f"Total memory usage per layer: {sum(mem_list)} MB")
#     return layer_outputs, mem_list


def model_memory_usage(x, model, device): 

    torch.cuda.empty_cache()

    layers_list = extract_layers(model.model.model) 

    mem_list = []

    layer_outputs = []

    for i, layer in enumerate(layers_list):
        if type(layer).__name__ == "Concat":
        
            if i == 11:
                concat_inputs = [layer_outputs[6], layer_outputs[10]]
            elif i == 14:
                concat_inputs = [layer_outputs[4], layer_outputs[13]]
            elif i == 17:
                concat_inputs = [layer_outputs[12], layer_outputs[16]]
            elif i == 20:
                concat_inputs = [layer_outputs[9], layer_outputs[19]]
        
            x, used_memory = measure_memory(concat_inputs, [layer], device)
        
            mem_list.append(used_memory)
            layer_outputs.append(x)

            setattr(layer, 'memory_usg', used_memory)

            #print(f" Layer {i} ({type(layer).__name__}) memory usage: {layer.memory_usg:.5f} MB")
        
        # print("concat_inputs")
        # print(concat_inputs)
        # print("x")
        # print(x)
        
            continue

################################################################## need to modify
        if type(layer).__name__ == "Detect":
            inputs = [layer_outputs[15], layer_outputs[18], layer_outputs[21]]

            x, used_memory = measure_memory(inputs, [layer], device)
            mem_list.append(used_memory)

            setattr(layer, 'memory_usg', used_memory)

            #print(f" Layer {i} ({type(layer).__name__}) memory usage: {layer.memory_usg:.5f} MB")

            continue
        
##################################################################        
        else:
            x, used_memory = measure_memory(x, [layer], device)
            mem_list.append(used_memory)
            layer_outputs.append(x)

            setattr(layer, 'memory_usg', used_memory)
        
            #print(f" Layer {i} ({type(layer).__name__}) memory usage: {layer.memory_usg:.5f} MB")

    #print(f"Total memory usage per layer: {sum(mem_list)} MB")
    return sum(mem_list)


def model_memory_usage_with_reducing(x, pruned_model, device):

    torch.cuda.empty_cache()

    reduced_model = copy.deepcopy(pruned_model)
    
    # ✅ DDP인지 확인하고 .module로 접근할지 .model로 접근할지 결정
    if isinstance(pruned_model, torch.nn.parallel.DistributedDataParallel):
        pruned_base_model = pruned_model.module
        reduced_base_model = reduced_model.module
    else:
        pruned_base_model = pruned_model.model
        reduced_base_model = reduced_model.model

    reduced_base_model = reduced_base_model.to(device).eval()
    pruned_base_model = pruned_base_model.to(device).eval()
    
    yolov8_reducing(pruned_base_model, reduced_base_model)

    reduced_layers = extract_layers(reduced_base_model)
    pruned_layers = extract_layers(pruned_base_model)

    mem_list = []
    layer_outputs = []

    x = x.to(device)
    
    for i, layer in enumerate(reduced_layers):
        layer = layer.to(device)
        
        if type(layer).__name__ == "Concat":
        
            if i == 11:
                concat_inputs = [layer_outputs[6], layer_outputs[10]]
            elif i == 14:
                concat_inputs = [layer_outputs[4], layer_outputs[13]]
            elif i == 17:
                concat_inputs = [layer_outputs[12], layer_outputs[16]]
            elif i == 20:
                concat_inputs = [layer_outputs[9], layer_outputs[19]]
                
            concat_inputs = [ci.to(device) for ci in concat_inputs]
            
            x, used_memory = measure_memory(concat_inputs, [layer], device)
        
            mem_list.append(used_memory)
            layer_outputs.append(x)

            setattr(pruned_layers[i], 'memory_usg', used_memory)
        
            continue

        if type(layer).__name__ == "Detect":
            inputs = [layer_outputs[15], layer_outputs[18], layer_outputs[21]]
            inputs = [inp.to(device) for inp in inputs]
            
            x, used_memory = measure_memory(inputs, [layer], device)
            mem_list.append(used_memory)

            setattr(pruned_layers[i], 'memory_usg', used_memory)

            continue
        
        else:
            x = x.to(device)
            x, used_memory = measure_memory(x, [layer], device)
            mem_list.append(used_memory)
            layer_outputs.append(x)

            setattr(pruned_layers[i], 'memory_usg', used_memory)
        
    #print(f"Total memory usage per layer: {sum(mem_list)} MB")

    return sum(mem_list)


def custom_memory_loss_function(memory, hyperparam, device_condition_memory):
    memory_diff = memory - device_condition_memory
    memory_loss = max(0, memory_diff)

    final_loss = hyperparam * memory_loss
    
    return torch.tensor(final_loss, requires_grad=True)
