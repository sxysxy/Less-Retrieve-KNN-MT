from inference import *

if __name__ == "__main__":
    ps = ArgumentParser()
    ps.add_argument("--dataset", required=True, choices=['it', 'koran', 'law', 'medical'])
    ps.add_argument("--single-gpu-index", default=0)
    ps.add_argument("--no-translation-loss", default=False, action='store_true')
    args = ps.parse_args()
    cmd = [
        get_dataset_path(args.dataset),
        "--task", "translation",
        "--train-subset", "valid", "--valid-subset", "valid",
        "--best-checkpoint-metric", "f1",
        "--finetune-from-model", BASE_MODEL,
        "--optimizer", "adam", "--adam-betas", "(0.9, 0.98)", "--adam-eps", "1e-8",
        "--lr", "1e-4", 
        "--lr-scheduler", "reduce_lr_on_plateau", 
        "--min-lr", "3e-05", "--lr-patience", "5", "--lr-shrink", "0.5", "--patience", "30", "--max-update", "5000", 
        "--max-epoch", "100",
        "--criterion", "less_retrieve_criterion", 
        "--save-interval-updates", "100", 
        "--no-epoch-checkpoints", "--no-last-checkpoints", "--no-save-optimizer-state",
        "--tensorboard-logdir", pjdir(f"save-models/LRKNNMT/{args.dataset}/log"),
        "--batch-size", "4",
        "--update-freq", "8",
        "--user-dir", pjdir("knnbox/models"),
        "--arch", "less_retrieve_knn_mt@transformer_wmt19_de_en",
        "--whether_retrieve_selector_path", 
            pjdir(f"save-models/LRKNNMT/{args.dataset}/selector.pt") if not args.no_translation_loss else pjdir(f"save-models/LRKNNMT/{args.dataset}/selector_no_translation_loss.pt"),
        "--knn-mode", "train_less_retrieve", 
        "--knn-datastore-path", pjdir(f"datastore/vanilla/{args.dataset}"),
        "--knn-k", "8",
        "--knn-lambda", str(DEFAULT_KNN_LAMBDA[args.dataset]),
        "--knn-temperature", str(DEFAULT_KNN_TEMPERATURE[args.dataset])
    ]
    
    if args.no_translation_loss:
        cmd.extend(["--use_mt_loss_for_selector", "False"])
    
    script = [sys.executable, pjdir("fairseq_cli/train.py")]
    script.extend(cmd)
    
    print(' '.join(script))    
    
    p = subprocess.Popen(script, env=get_base_env(args))
    p.wait()
    exit(p.returncode)