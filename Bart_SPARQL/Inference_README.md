# Inference BART Program


## Setup

### Download Data and Models
    wget https://cloud.tsinghua.edu.cn/f/1b9746dcd96b4fca870d/?dl=1
    mv 'index.html?dl=1' bart_program_ckpt.zip
    unzip bart_program_ckpt.zip
    wget https://cloud.tsinghua.edu.cn/f/26d05edaa5d0480bb3ae/?dl=1
    mv 'index.html?dl=1' kqa_pro.zip
    unzip kqa_pro.zip
    
### Clone Inference Code

    git clone https://github.com/Ankur3107/KQAPro_Baselines
    cd KQAPro_Baselines

### Install dependencies
    pip install -q transformers sentencepiece
    
## Run Inference
    from Bart_SPARQL.inference import Inference
    model_name_or_path = "../KQAPro_ckpt/sparql_ckpt"
    kb_json_file = "../KQA-Pro-v1.0/kb.json"
    inferencer = Inference(model_name_or_path, kb_json_file)
    inferencer.run("who is the prime minister of india?")
