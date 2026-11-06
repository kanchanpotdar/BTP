"""Build the self-contained Kaggle LoRA notebook from the reviewed F1 trainer."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "scripts/phase7_f1_train.py").read_text()
source = source.replace("#!/usr/bin/env python3\n", "", 1)
source = source[: source.index('\nif __name__ == "__main__":')]


def replace(old, new):
    global source
    assert source.count(old) == 1, (old[:100], source.count(old))
    source = source.replace(old, new)


replace('    parser.add_argument("--learning-rate", type=float, default=1e-5)',
        '    parser.add_argument("--learning-rate", type=float, default=1e-4)\n'
        '    parser.add_argument("--lora-rank", type=int, default=8)\n'
        '    parser.add_argument("--lora-alpha", type=int, default=16)\n'
        '    parser.add_argument("--lora-dropout", type=float, default=0.05)\n'
        '    parser.add_argument("--probe-only", action="store_true")')
replace('    assert args.validation_batch == 8',
        '    assert args.validation_batch == 8\n'
        '    assert (args.lora_rank, args.lora_alpha) in ((8, 16), (16, 32))\n'
        '    assert args.lora_dropout == 0.05 and args.learning_rate in (1e-4, 2e-5)')
replace('    model_source = args.resume / "model" if args.resume else MODEL_ID\n'
        '    model_kwargs = {"dtype": torch.float32, "attn_implementation": "sdpa"}\n'
        '    if not args.resume:\n'
        '        model_kwargs["revision"] = MODEL_REVISION\n'
        '    model = WhisperForConditionalGeneration.from_pretrained(model_source, **model_kwargs).to(device).train()\n'
        '    model.config.use_cache = False',
        '    from peft import LoraConfig, PeftModel, get_peft_model\n'
        '    base = WhisperForConditionalGeneration.from_pretrained(\n'
        '        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, attn_implementation="sdpa"\n'
        '    )\n'
        '    base.config.use_cache = False\n'
        '    targets = [(name, type(module).__name__) for name, module in base.named_modules()\n'
        '               if name.endswith(("q_proj", "v_proj"))]\n'
        '    assert targets and any("encoder" in name for name, _ in targets)\n'
        '    assert any("decoder" in name for name, _ in targets)\n'
        '    assert all(kind == "Linear" for _, kind in targets), targets\n'
        '    print("LoRA targets:", targets)\n'
        '    if args.resume:\n'
        '        model = PeftModel.from_pretrained(\n'
        '            base, args.resume / "adapter", is_trainable=True\n'
        '        )\n'
        '    else:\n'
        '        lora = LoraConfig(\n'
        '            r=args.lora_rank, lora_alpha=args.lora_alpha,\n'
        '            lora_dropout=args.lora_dropout, target_modules=["q_proj", "v_proj"],\n'
        '            bias="none",\n'
        '        )\n'
        '        model = get_peft_model(base, lora)\n'
        '    model = model.to(device).train()\n'
        '    model.config.use_cache = False\n'
        '    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n'
        '    total = sum(p.numel() for p in model.parameters())\n'
        '    assert 0 < trainable < total / 10\n'
        '    assert all(("lora_" in name) == parameter.requires_grad\n'
        '               for name, parameter in model.named_parameters())\n'
        '    print({"trainable": trainable, "total": total, "percent": 100*trainable/total})')
replace('        "patience": args.patience, "learning_rate": args.learning_rate,',
        '        "patience": args.patience, "learning_rate": args.learning_rate,\n'
        '        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,\n'
        '        "lora_dropout": args.lora_dropout, "peft_version": __import__("peft").__version__,')
replace('    run_started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)',
        '    if args.probe_only:\n'
        '        # One real forward/backward/update; restore by loading the saved adapter.\n'
        '        # The full-run warmup scheduler initializes optimizer LR to zero.\n'
        '        # Give this isolated probe a nonzero LR so a successful step\n'
        '        # must actually change an adapter weight. Full runs keep warmup.\n'
        '        for group in optimizer.param_groups:\n'
        '            group["lr"] = args.learning_rate\n'
        '        assert all(group["lr"] > 0 for group in optimizer.param_groups)\n'
        '        rows = train_rows[:args.physical_batch]\n'
        '        features = make_features(train_root, rows).to(device)\n'
        '        labels = make_labels(rows).to(device)\n'
        '        # FP16 may overflow at the initial scale. Retry exactly this batch\n'
        '        # with the reduced scaler value; require an actual optimizer step.\n'
        '        for attempt in range(8):\n'
        '            optimizer.zero_grad(set_to_none=True)\n'
        '            with torch.autocast(device_type="cuda", dtype=torch.float16):\n'
        '                loss = model(input_features=features, labels=labels).loss\n'
        '            assert torch.isfinite(loss).item(), "Nonfinite probe loss"\n'
        '            scaler.scale(loss).backward()\n'
        '            scaler.unscale_(optimizer)\n'
        '            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]\n'
        '            if not grads or not all(torch.isfinite(g).all().item() for g in grads):\n'
        '                old_scale = scaler.get_scale()\n'
        '                scaler.step(optimizer); scaler.update()\n'
        '                assert scaler.get_scale() < old_scale, "AMP did not reduce on overflow"\n'
        '                continue\n'
        '            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)\n'
        '            assert torch.isfinite(norm).item() and float(norm) > 0\n'
        '            changed = [(p, p.detach().clone()) for p in model.parameters()\n'
        '                       if p.requires_grad and p.grad is not None and torch.count_nonzero(p.grad).item()]\n'
        '            assert changed, "No nonzero LoRA gradients"\n'
        '            old_scale = scaler.get_scale()\n'
        '            scaler.step(optimizer); scaler.update()\n'
        '            if scaler.get_scale() < old_scale:\n'
        '                continue\n'
        '            assert any(not torch.equal(before, p.detach()) for p, before in changed), "No LoRA weight changed after a finite optimizer step"\n'
        '            break\n'
        '        else:\n'
        '            raise RuntimeError("Probe could not complete a finite update after 8 AMP retries")\n'
        '        gate_path = args.output / "probe_adapter"\n'
        '        model.save_pretrained(gate_path, safe_serialization=True)\n'
        '        del model, optimizer, features, labels, grads, changed\n'
        '        gc.collect(); torch.cuda.empty_cache()\n'
        '        base2 = WhisperForConditionalGeneration.from_pretrained(\n'
        '            MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32\n'
        '        )\n'
        '        restored = PeftModel.from_pretrained(base2, gate_path, is_trainable=True)\n'
        '        assert any(p.requires_grad for p in restored.parameters())\n'
        '        print({"probe": "PASS", "loss": float(loss), "grad_norm": float(norm),\n'
        '               "adapter_reload": True, "peak_mib": torch.cuda.max_memory_allocated(device)/2**20})\n'
        '        return\n'
        '    run_started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)')
replace('checkpoint / "model"', 'checkpoint / "adapter"')
replace('"experiment": "F1", "status": "complete",',
        '"experiment": "L1 LoRA", "status": "complete", "lora_rank": args.lora_rank,\n'
        '              "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,\n'
        '              "trainable_parameters": trainable, "total_parameters": total,\n'
        '              "target_modules": targets, "peft_version": __import__("peft").__version__,')
replace('"learning_rate": args.learning_rate, "repetition_penalty":',
        '"learning_rate": args.learning_rate, "repetition_penalty":')
source = source.replace('F1_report_private.json', 'L1_report_private.json')
source = source.replace('print("F1 complete; best epoch"', 'print("L1 complete; best epoch"')
source = source.replace('skip_special_tokens=False) == reference',
                        'skip_special_tokens=False, clean_up_tokenization_spaces=False) == reference')
source = source.replace('skip_special_tokens=True); hypothesis',
                        'skip_special_tokens=True, clean_up_tokenization_spaces=False); hypothesis')


def markdown(value):
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(keepends=True)}


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": value.splitlines(keepends=True)}


cells = [
    markdown("""# Phase 7 L1 LoRA comparison on Kaggle

Run both configurations in this one self-contained notebook. Each run starts independently from pinned `neurlang/ipa-whisper-base` and uses all 3,629 frozen training clips. Validation uses all 732 clips, the same IPA normalization/CER, and repetition penalty 1.08. The 683-clip test set remains sealed. Compare against full fine-tuning `2e-5`, Epoch 5, CER 47.23950%.

Attach only the private `phase7_f1_train_v1` and `phase5_validation_baseline` datasets. Enable a T4 GPU and Internet. Do not attach model checkpoints or test data.
"""),
    code("""# Pin compatible package versions and remove Kaggle's incompatible optional torchao.
# This experiment uses ordinary FP16 LoRA, not torchao quantization.
import importlib.metadata as md, subprocess, sys
from packaging.version import Version
PINS = {'transformers':'4.57.1', 'tokenizers':'0.22.2',
        'huggingface-hub':'0.36.2', 'safetensors':'0.8.0', 'peft':'0.18.0'}
installed = {}
for name in PINS:
    try: installed[name] = md.version(name)
    except md.PackageNotFoundError: installed[name] = None
print(installed)
changed = False
# PEFT 0.18 calls its torchao dispatcher even for ordinary Linear layers.
# torchao below 0.16 raises during adapter injection, so remove it if present.
try: torchao_version = md.version('torchao')
except md.PackageNotFoundError: torchao_version = None
if torchao_version is not None and Version(torchao_version) < Version('0.16.0'):
    print(f'Removing incompatible optional torchao {torchao_version}')
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'], check=True)
    changed = True
if installed != PINS:
    subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir',
                    *[f'{name}=={version}' for name,version in PINS.items()]], check=True)
    changed = True
if changed:
    raise RuntimeError('Restart the Kaggle kernel, then rerun every cell.')
# Fail before loading the model if an incompatible optional package remains.
if torchao_version is not None:
    assert Version(torchao_version) >= Version('0.16.0')
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
"""),
    markdown("""## Embedded trainer and engineering gate

This code reuses the F1 bundle, feature, target, optimizer, scoring, and checkpoint contract. PEFT adds trainable `q_proj`/`v_proj` adapters in encoder and decoder. The probe checks real gradients, one optimizer update, adapter save, and trainable reload before the full experiment. Progress bars display training updates and validation batches.
"""),
    code(source + "\n"),
    code("""# Discover the frozen inputs by bundle marker and keep experiments separate.
from pathlib import Path
import json, gc
ROOT = Path('/kaggle/working/phase7_L1_private')
TRAIN = discover_bundle(Path('/kaggle/input'), 'phase7_f1_train_v1')
VAL = discover_bundle(Path('/kaggle/input'), 'phase5_validation_baseline')
ROOT.mkdir(parents=True, exist_ok=True)
print({'training':str(TRAIN), 'validation':str(VAL)})

def args_for(rank, output, extra=None, lr=1e-4):
    return ['--train-input',str(TRAIN),'--validation-input',str(VAL),
            '--output',str(output),'--lora-rank',str(rank),
            '--lora-alpha',str(2*rank),'--lora-dropout','0.05',
            '--learning-rate',str(lr),'--repetition-penalty','1.08',
            '--physical-batch','2','--accumulation','4','--validation-batch','8',
            '--max-epochs','10','--patience','3','--seed','0'] + (extra or [])

# Run the small engineering gate once; this does not enter either comparison history.
main(args_for(8, ROOT/'engineering_gate', ['--probe-only']))
"""),
    markdown("""## L1 baseline — rank 8, alpha 16, dropout 0.05

This is the declared sprint-plan configuration. If interrupted, rerun this cell: it resumes from the latest complete epoch checkpoint. A partially completed epoch runs again.
"""),
    code("""# Independent pretrained initialization; resume only within this rank-8 run.
out8 = ROOT/'rank8_alpha16'
report8 = out8/'L1_report_private.json'
if report8.exists():
    print('Rank 8 already complete:', json.loads(report8.read_text())['best_validation_cer'])
else:
    prior = sorted(p for p in out8.glob('epoch_*') if (p/'adapter'/'adapter_config.json').is_file() and (p/'training_state.pt').is_file())
    main(args_for(8, out8, ['--resume',str(prior[-1])] if prior else []))
"""),
    markdown("""## Learning-rate control — rank 8 at 2e-5

This run uses the successful full fine-tuning learning rate with the same rank 8 adapter. It isolates the learning-rate choice within LoRA; the full fine-tuning score remains a separate reference.
"""),
    code("""# Same rank and alpha as L1 baseline; only the learning rate changes.
out8low = ROOT/'rank8_alpha16_lr2em5'
report8low = out8low/'L1_report_private.json'
if report8low.exists():
    print('Rank 8 / 2e-5 complete:', json.loads(report8low.read_text())['best_validation_cer'])
else:
    prior = sorted(p for p in out8low.glob('epoch_*') if (p/'adapter'/'adapter_config.json').is_file() and (p/'training_state.pt').is_file())
    main(args_for(8, out8low, ['--resume',str(prior[-1])] if prior else [], lr=2e-5))
"""),
    markdown("""## Capacity check — rank 16, alpha 32, dropout 0.05

Only adapter rank/alpha change; alpha-to-rank ratio remains two. This is a second full-data run, independently initialized from the original pinned model.
"""),
    code("""# Independent pretrained initialization; resume only within this rank-16 run.
out16 = ROOT/'rank16_alpha32'
report16 = out16/'L1_report_private.json'
if report16.exists():
    print('Rank 16 already complete:', json.loads(report16.read_text())['best_validation_cer'])
else:
    prior = sorted(p for p in out16.glob('epoch_*') if (p/'adapter'/'adapter_config.json').is_file() and (p/'training_state.pt').is_file())
    main(args_for(16, out16, ['--resume',str(prior[-1])] if prior else []))
"""),
    markdown("""## Compare validation quality and resources

The full fine-tuning line is the previously measured `2e-5`, Epoch 5 result. New LoRA reports use the same 732 validation clips and 64,137 reference code points. Select by minimum corpus CER, then examine edits, training time, and memory.
"""),
    code("""# Load aggregate histories and render publication-friendly figures.
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
run_dirs = {'r8 1e-4':'rank8_alpha16', 'r8 2e-5':'rank8_alpha16_lr2em5',
            'r16 1e-4':'rank16_alpha32'}
reports = {label:json.loads((ROOT/folder/'L1_report_private.json').read_text())
           for label,folder in run_dirs.items()}
rows, summaries = [], []
for label, report in reports.items():
    rank = report['lora_rank']
    assert report['test_data_used'] is False and report['train_clips']==3629 and report['validation_clips']==732
    assert report['lora_rank']==rank and report['repetition_penalty']==1.08
    for epoch in report['history']:
        v=epoch['validation']
        assert v['clips']==732 and v['reference_codepoints']==64137
        rows.append({'run':label,'rank':rank,'epoch':epoch['epoch'],'loss':epoch['training_loss'],
                     'CER_percent':100*v['corpus_cer'],'S':v['S'],'D':v['D'],'I':v['I'],
                     'EOS':v['eos_count'],'cap_hits':v['cap_without_eos'],
                     'minutes':epoch['epoch_seconds']/60,'peak_MiB':epoch['train_peak_allocated_mib']})
    best=next(x for x in report['history'] if x['epoch']==report['best_epoch'])['validation']
    summaries.append({'run':label,'rank':rank,'learning_rate':report['learning_rate'],
                      'best_epoch':report['best_epoch'],
                      'CER_percent':100*report['best_validation_cer'],'S':best['S'],'D':best['D'],'I':best['I'],
                      'EOS':best['eos_count'],'cap_hits':best['cap_without_eos'],
                      'trainable':report['trainable_parameters'],
                      'minutes':sum(x['epoch_seconds'] for x in report['history'])/60,
                      'peak_MiB':report['peak_allocated_mib']})
epochs=pd.DataFrame(rows); summary=pd.DataFrame(summaries)
display(epochs.round(3)); display(summary.round(3))
print('Full F1 reference: 2e-5, Epoch 5, 47.23950% CER, S/D/I 14503/5583/10212')
PLOTS=ROOT/'plots'; PLOTS.mkdir(exist_ok=True)
fig, axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
for label,g in epochs.groupby('run'):
    axes[0,0].plot(g.epoch,g.CER_percent,'o-',label=label)
    axes[0,1].plot(g.epoch,g.loss,'o-',label=label)
axes[0,0].axhline(47.239502939,ls='--',color='black',label='Full F1 r=none')
axes[0,0].set(xlabel='Epoch',ylabel='Validation CER (%)',title='Validation quality')
axes[0,1].set(xlabel='Epoch',ylabel='Training loss',title='Training trajectory')
bottom=[0]*len(summary)
for key,color in [('S','#3b82f6'),('D','#f59e0b'),('I','#ef4444')]:
    axes[1,0].bar(summary['run'],summary[key],bottom=bottom,label=key,color=color)
    bottom=[a+b for a,b in zip(bottom,summary[key])]
axes[1,0].set(ylabel='Edit count',title='Best checkpoint edits')
axes[1,1].bar(summary['run'],summary['peak_MiB'],color=['#16a34a','#0ea5e9','#7c3aed'])
axes[1,1].set(ylabel='Peak allocated MiB',title='GPU memory')
for ax in axes.flat:
    if ax==axes[1,0] or ax==axes[0,0] or ax==axes[0,1]: ax.legend()
fig.savefig(PLOTS/'lora_comparison.png',dpi=220,bbox_inches='tight')
plt.show()
epochs.to_csv(PLOTS/'epoch_metrics.csv',index=False)
summary.to_csv(PLOTS/'best_summary.csv',index=False)
"""),
    markdown("""## Download results and selected adapters

The results ZIP includes aggregate reports, private prediction JSONL files, and plots. Each adapter ZIP contains the selected adapter plus optimizer, scheduler, scaler, and RNG state for recovery. Keep these archives private.
"""),
    code("""# Stream ZIP members from disk; avoid loading large checkpoints into memory.
import zipfile
from tqdm.auto import tqdm
from IPython.display import FileLink, display
results_zip=Path('/kaggle/working/phase7_L1_results_private.zip')
files=list(PLOTS.glob('*'))
for folder_name in run_dirs.values():
    folder=ROOT/folder_name
    files += [folder/'L1_report_private.json',folder/'aggregate_history.json']
    files += list(folder.glob('validation_epoch_*_private.jsonl'))
with zipfile.ZipFile(results_zip,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for path in tqdm(files,desc='Zipping results',unit='file'):
        assert path.is_file()
        z.write(path,path.relative_to(ROOT))
archives=[results_zip]
for label,report in reports.items():
    best=ROOT/run_dirs[label]/f"epoch_{report['best_epoch']:02d}"
    assert (best/'adapter'/'adapter_config.json').is_file() and (best/'training_state.pt').is_file()
    safe_label=label.replace(' ','_').replace('-','m')
    dest=Path(f'/kaggle/working/phase7_L1_{safe_label}_best_adapter_private.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in tqdm([p for p in best.rglob('*') if p.is_file()],desc=f'{label} checkpoint',unit='file'):
            z.write(path,path.relative_to(best.parent))
    archives.append(dest)
for archive in archives:
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    print(archive.name,round(archive.stat().st_size/2**20,2),'MiB')
    display(FileLink(str(archive)))
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks/Phase-7-L1-LoRA-Self-Contained-Kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
print(out)
