# ntt-rc-gpu-simulation

## プログラムに関して

### 単一の訓練時のサンプルプログラム

- 下記コマンドにより実行
    - モデル等の設定は`config.yml`を使用して指定
    - `config.yml`の設定に関する情報は内部のコメントを参照

```bash
python main.py --config config.yml
```

### 複数実験を回す場合のサンプルプログラム

- 使用する実験ハイパラ等の設定は`experiment.py`内の`params`を変更して指定
    - `params`で指定されなかった項目は`config.yml`で指定されたものが使用される

```bash
python3 experiment.py --config config.yml
```

### Stable Diffusionに関するサンプルプログラム

- 事前学習済みのモデルを`timm`等により読み込み、`analog_convert`関数を使用することで`Linear`や`Conv2d`を自動置換可能

```bash
$ cd examples
$ python3 stable_diffusion.py
$ python3 stable_diffusion_viz.py
```

- CLIP Score, FID評価のためのサンプルプログラムも格納している

```bash
$ cd examples
$ python3 stable_diffusion_clip_score.py
$ python3 stable_diffusion_fid.py
```

### LoRAに関するサンプルプログラム

- 事前学習済みのモデルを読み込み、`peft`によりLoRA評価を行うサンプル
    - `analog_convert`関数を使用することで`Linear`や`Conv2d`を自動置換可能

```bash
$ cd examples
$ python3 vit_lora.py
$ python3 vit_lora_analog.py
```

## configの指定に関する一例

### Quantization-Aware Trainingの場合

- `train_mode`を`analog`にすることでアナログ版`forward`が走るためQATとなる

```yml
train_mode: analog
epochs: 30
```

### Post-Training Quantizationの場合

- epochsを0にすることで、読み込まれた直後のモデルの推論が一度走り終了する
- `train_mode`を`analog`にすることでアナログ版`forward`が走るため、評価はPTQとなる

```yml
train_mode: analog
epochs: 0
```

### PTQのための事前学習の場合

- `train_mode`を`digital`にすることで通常の`Linear`のforawrdが走る
    - PTQをする場合は、保存したモデルを読み込むように指定して、上記の設定を行う

```yml
train_mode: digital
epochs: 30
```

## その他

- 大量のCPUコアを持つ計算機で`Matmul`の計算をCPUで行う場合、逆に遅くなる場合が多い
    - 下記の指定をすることで、高速に実行できるようになる

```bash
export OMP_NUM_THREADS=4
```
