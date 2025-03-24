from argparse import ArgumentParser
import os
import subprocess
import argparse
import sys
import re
from datetime import datetime
PROJECT_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
BASE_MODEL = f"{PROJECT_PATH}/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt"
ARCH_SUFFIX = "transformer_wmt19_de_en"

DEFAULT_DATASTORE_PATH = {
    ("vanilla", "it") : f"{PROJECT_PATH}/datastore/vanilla/it",
    ("vanilla", "koran") : f"{PROJECT_PATH}/datastore/vanilla/koran",
    ("vanilla", "law") : f"{PROJECT_PATH}/datastore/vanilla/law",
    ("vanilla", "medical") : f"{PROJECT_PATH}/datastore/vanilla/medical",
    ("adaptive", "it") : f"{PROJECT_PATH}/datastore/vanilla/it",
    ("adaptive", "koran") : f"{PROJECT_PATH}/datastore/vanilla/koran",
    ("adaptive", "law") : f"{PROJECT_PATH}/datastore/vanilla/law",
    ("adaptive", "medical") : f"{PROJECT_PATH}/datastore/vanilla/medical",
    ("pck", "it") : f"{PROJECT_PATH}/datastore/pck/it_dim64",
    ("pck", "koran") : f"{PROJECT_PATH}/datastore/pck/koran_dim64",
    ("pck", "law") : f"{PROJECT_PATH}/datastore/pck/law_dim64",
    ("pck", "medical") : f"{PROJECT_PATH}/datastore/pck/medical_dim64",
    ("lr", "it") : f"{PROJECT_PATH}/datastore/vanilla/it",
    ("lr", "koran") : f"{PROJECT_PATH}/datastore/vanilla/koran",
    ("lr", "law") : f"{PROJECT_PATH}/datastore/vanilla/law",
    ("lr", "medical") : f"{PROJECT_PATH}/datastore/vanilla/medical",
    ("lr_adaptive", "it") :  f"{PROJECT_PATH}/datastore/vanilla/it",
    ("lr_adaptive", "koran") : f"{PROJECT_PATH}/datastore/vanilla/koran",
    ("lr_adaptive", "law") : f"{PROJECT_PATH}/datastore/vanilla/law",
    ("lr_adaptive", "medical") : f"{PROJECT_PATH}/datastore/vanilla/medical",
    ("lr_pck", "it") :  f"{PROJECT_PATH}/datastore/pck/it_dim64",
    ("lr_pck", "koran") : f"{PROJECT_PATH}/datastore/pck/koran_dim64",
    ("lr_pck", "law") : f"{PROJECT_PATH}/datastore/pck/law_dim64",
    ("lr_pck", "medical") : f"{PROJECT_PATH}/datastore/pck/medical_dim64"
}

DEFAULT_KNN_TEMPERATURE = {
    "it" : 10,
    "law" : 10,
    "koran" : 100,
    "medical" : 10   
}

DEFAULT_KNN_LAMBDA = {
    "it" : 0.7,
    "medical" : 0.8,
    "law" : 0.8,
    "koran" : 0.8
}

def get_base_path():
    return f"{PROJECT_PATH}/data-bin"

def get_dataset_path(ds):
    return f"{PROJECT_PATH}/data-bin/{ds}"

def get_dstore_path(method, ds):
    return f"{PROJECT_PATH}/datastore/{method}/ds"

def pjdir(x):
    return os.path.join(PROJECT_PATH, x)

def get_base_env(args):
    e = os.environ.copy()
    e["OMP_WAIT_POLICY"] = "PASSIVE"
    e["CUDA_VISIBLE_DEVICES"] = str(args.single_gpu_index)
    return e

def get_arch(m):
    if m == 'lr':
        return f"less_retrieve_knn_mt@{ARCH_SUFFIX}"
    else:
        return f"{m}_knn_mt@{ARCH_SUFFIX}"
    
def add_common_arguments(parser : argparse.ArgumentParser):
    #parser.add_argument("--model", required=True, choices=['base', 'vanilla', 'adaptive', 'pck', 'lr', 'lr_adaptive', 'lr_pck'])
    parser.add_argument("--dataset", required=True, choices=['it', 'koran', 'law', 'medical'])
    parser.add_argument("--single-gpu-index", default=0)
    parser.add_argument("--run-3-time", default=False, action='store_true')
    parser.add_argument("--no-translation-loss", default=False, action='store_true')
    parser.add_argument("--test-knn-overhead", default=False, action='store_true')
    parser.add_argument("--router-threshold", default=0.5, type=float)
    


if __name__ == "__main__":
    ps = ArgumentParser()
    add_common_arguments(ps)
    args = ps.parse_args()
 
    cmd = [ 
        get_dataset_path(args.dataset),
        "--task", "translation_with_moe",
        "--path", BASE_MODEL,
        "--dataset-impl", "mmap",
        "--beam", "4",
        "--lenpen", "0.6",
        "--max-len-a", "1.2",
        "--max-len-b", "10",
        "--source-lang", "de", 
        "--target-lang", "en", 
        "--gen-subset", "test",
        "--max-tokens", "2048", 
        "--scoring", "sacrebleu", 
        "--tokenizer", "moses",
        "--remove-bpe",
        "--user-dir", pjdir("knnbox/models"),
        "--arch", "moe_knn_mt@transformer_wmt19_de_en",
        "--knn-mode", "inference",
        "--knn-datastore-path", '0',
        "--inference-router-threshold", str(args.router_threshold)
    ]    
    script = [sys.executable, pjdir("knnbox-scripts/common/generate.py")]
    script.extend(cmd)
    
    print(' '.join(script))    
    
    if not args.run_3_time:
        p = subprocess.Popen(script, env=get_base_env(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        p.wait()
        if p.returncode != 0:
            print(f"Error:\n {err.decode()}")
            exit(p.returncode)
        else:
            speed_info = err.decode().split("|")[-1].strip()
            print(speed_info)
            bleu_info = out.decode().split("Generate test with")[-1].strip()
            print(bleu_info)

            inference_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_results")
            os.makedirs(inference_results_path, mode=0o755, exist_ok=True)
            with open(f"{inference_results_path}/{args.dataset}-{datetime.now()}.txt", "w") as f:
                f.write(f"{speed_info}\n{bleu_info}")
                
                if args.no_translation_loss:
                    f.write("\nNote: The selector was trained without translation loss.")
            
            exit(0)
    else:
        bleu = None
        times = []
        tps = []
        knn_time = []
        for li in range(3):
            p = subprocess.Popen(script, env=get_base_env(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            p.wait()
            
            if p.returncode != 0:
                print(f"Error:\n {err.decode()}")
                exit(p.returncode)
            else:
                out_info = out.decode()
                
                if bleu is None:
                    #import pdb
                    #pdb.set_trace()
                    bleu_info = out_info.split("\n")[-3].split("Generate test with")[-1].strip()
                   # bleu = float(re.match(r"BLEU = ([\d\.]+)", bleu_info)[1])
                    bleu = float(bleu_info.split(":")[1].split(' ')[3])
                    print(f"BLEU = {bleu}")
                    
                if args.test_knn_overhead:
                    knn_retrieve_info = out_info.split("\n")[-2]
                    knn_time.append( float(knn_retrieve_info.split('=')[-1].strip().split('s')[0]) )                
            
                speed_info = err.decode().split("|")[-1].strip()
                #print(speed_info)
                times.append(float(speed_info.split('in')[1].split(' ')[1].split('s')[0]))
                tps.append(float(speed_info.split("/s")[-2].split(' ')[-2]))
                if args.test_knn_overhead:
                    print(f"Run #{li+1}, inference time = {times[-1]}, {tps[-1]} tokens/s, KNN overhead = {knn_time[-1]}s")
                else:
                    print(f"Run #{li+1}, inference time = {times[-1]}, {tps[-1]} tokens/s")
                
                

        inference_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_results")
        os.makedirs(inference_results_path, mode=0o755, exist_ok=True)
        with open(f"{inference_results_path}/{args.dataset}-{datetime.now()}.txt", "w") as f:
            f.write(f"BLEU = {bleu}\nEach inference time in seconds = {' '.join(list(map(str,times)))}\nAverage inference time = {sum(times)/len(times)}s")
            f.write(f"\nEach inference speed(tokens/s) = {' '.join(list(map(str, tps)))}\nAverage inference speed = {sum(tps)/len(tps)} tokens/s")
            
            if args.test_knn_overhead:
                f.write(f"\nEach KNN retrieving time = {' '.join(list(map(str, knn_time)))}\nAverage KNN retrieving time = {sum(knn_time)/len(knn_time)}s")
            
        exit(0)

            