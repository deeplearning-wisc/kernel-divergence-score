

model_dirs = {
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'llama3.1': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi3-small': 'microsoft/Phi-3-small-128k-instruct'
}


def load_model(args, model_name=None, peft_path=None):
    model_name = args.model if model_name is None else model_name
    
    if model_name == 'mistral' :
        from model.mistral import MistralWrapper
        return MistralWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name in ['llama3.1', 'llama2-70b-chat', 'llama2-13b-chat', 'llama2-7b-chat', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b', ] :
        from model.llama import LlamaWrapper
        lversion = None
        if model_name in ['llama3.1', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b']:
            lversion = 3
        elif model_name in ['llama2-70b-chat', 'llama2-13b-chat', 'llama2-7b-chat']:
            lversion = 2
        return LlamaWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path, llama_version=lversion)
    elif model_name in ["phi3-small", "phi3-medium"] :
        from model.phi import PhiWrapper
        return PhiWrapper(args, model_dirs[model_name], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    else:
        raise ValueError("invalid model!")