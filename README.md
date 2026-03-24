# SpecSeg Refactor (HF-like bundles + model registry)

## Install
This is plain Python. You need: numpy, torch, scikit-learn, matplotlib, pyyaml.

## Train
Edit a config in `experiments/` then run:

```bash
python train.py --config experiments/focal_hier_linear.yaml
```

Outputs a HF-like folder:

- `config.json` (architecture name + kwargs)
- `stats.json` (mean/std)
- `model.pt` (state_dict)
- `loss_curve.png`

## Add a new architecture
1. Create `src/models/my_model.py` and decorate the class with `@register_model("my_name")`
2. Import it once in `train.py` (or add an auto-import mechanism)
3. Point your experiment config `model.name` to `my_name`

## Notes on AST / PANNs
Your notebooks include AST / CNN14 feature models; those are left as stubs here because they depend on external pretrained code.





# In practice

1. Define a model architecture in  a file from the folder src/models