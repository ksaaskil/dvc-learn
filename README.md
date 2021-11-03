# DVC learning

## Commands

### Tracking data

Track data with DVC:

```bash
$ dvc add data/data.xml
```

Note how `data/data.xml.dvc` file contains a "reference" to the data. This file should be version-controlled so it can be pulled with `dvc pull`.

If you make changes to the data, you should run `dvc add` again to update the reference.

### Working with remotes

Add a new remote for hosting data:

```bash
$ dvc remote add -d storage s3://mybucket/dvcstore
$ git add .dvc/config
```

Push data to remote:

```bash
$ dvc push
```

Pull data from remote:

```bash
$ dvc pull
```

### Data pipelines

Run data preparation pipeline:

```bash
$ dvc run -n prepare \
          -p prepare.seed,prepare.split \
          -d src/prepare.py -d data/data.xml \
          -o data/prepared \
          python src/prepare.py data/data.xml
```

This creates `data/prepared/train.tsv`. and `data/prepared/test.tsv`.

See `params.yaml` for parameters, `dvc.yaml` for DAG definition, and `dvc.lock` for the exact definition of what was done. The command `dvc repro` used for reproducing runs relies on the DAG definition from `dvc.yaml`, and uses `dvc.lock` to determine what exactly needs to be run.

Run feature generation pipeline:

```bash
$ dvc run -n featurize \
          -p featurize.max_features,featurize.ngrams \
          -d src/featurization.py -d data/prepared \
          -o data/features \
          python src/featurization.py data/prepared data/features
```

Run training step:

```bash
$ dvc run -n train \
          -p train.seed,train.n_est,train.min_split \
          -d src/train.py -d data/features \
          -o model.pkl \
          python src/train.py data/features model.pkl
```

Reproduce pipeline from `dvc.yaml`:

```bash
$ dvc repro
```

Visualize the pipeline:

```bash
$ dvc dag
```
