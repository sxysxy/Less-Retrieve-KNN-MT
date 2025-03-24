from inference import *

if __name__ == "__main__":
    ps = ArgumentParser()
    #ps.add_argument("--dataset", required=True, choices=['it', 'koran', 'law', 'medical'])
    ps.add_argument("--single-gpu-index", default=0)
    ps.add_argument("--max-epoch", default=50, type=int)
    #ps.add_argument("--no-translation-loss", default=False, action='store_true')
    args = ps.parse_args()
    cmd = [
        get_base_path(),
        "--task", "translation_with_moe",
        "--train-subset", "valid", 
        "--valid-subset", "test",
        "--best-checkpoint-metric", "loss",
        "--finetune-from-model", BASE_MODEL,
        "--optimizer", "adam", "--adam-betas", "(0.9, 0.98)", "--adam-eps", "1e-8",
        "--lr", "1e-4", 
        "--lr-scheduler", "reduce_lr_on_plateau", 
        "--min-lr", "3e-05", "--lr-patience", "5", "--lr-shrink", "0.5", "--patience", "30", "--max-update", "5000", 
        "--max-epoch", str(args.max_epoch),
        "--criterion", "moe_criterion", 
        "--save-interval-updates", "100", 
        "--no-epoch-checkpoints", "--no-last-checkpoints", "--no-save-optimizer-state",
        "--tensorboard-logdir", pjdir(f"save-models/MOEKNNMT/log"),
        "--batch-size", "10",
        "--update-freq", "8",
        "--user-dir", pjdir("knnbox/models"),
        "--arch", "moe_knn_mt@transformer_wmt19_de_en",
        # "--whether_retrieve_selector_path", 
        #     pjdir(f"save-models/LRKNNMT/{args.dataset}/selector.pt") if not args.no_translation_loss else pjdir(f"save-models/LRKNNMT/{args.dataset}/selector_no_translation_loss.pt"),
        "--knn-mode", "train", 
        "--knn-datastore-path", "1",
        "--knn-k", "8"
    ]

    script = [sys.executable, pjdir("fairseq_cli/train.py")]
    script.extend(cmd)
    
    print(' '.join(script))    
    
    p = subprocess.Popen(script, env=get_base_env(args))
    p.wait()
    exit(p.returncode)