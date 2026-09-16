# glm53-flash-nvfp4-2node

最新リリース: **v1**(git タグ `v1`)。再量子化したチェックポイントは
Hugging Face で `tenhkspark/GLM-5.3-Flash-NVFP4-Wabi` として公開
しています——同一リポジトリに git タグ `v1` と `v2`、最新版は
このファイルの先頭が常に示します。

`GLM-5.3-Flash-NVFP4` を二台の NVIDIA DGX Spark ノードで Ray テンソル
並列(TP=2)によりサービングする私のレシピと、シングルストリームの
デコード速度を取り戻すために使うウェイトオンリー再量子化。ここに
あるものはすべて私自身の実装と、私自身のハードウェアでの実測値
です——*不完全を楽しむ*。

モデルの重みは含みません。入力チェックポイントは NVIDIA の
`GLM-5.3-Flash-NVFP4` リリースです。再量子化スクリプトは何かを
再配布する代わりに、そのチェックポイントの BF16 dense linear を
書き換えます。スクリプトは各自のコピーに向けてください。

まだ改善のアイデアはいくつも残っていますが、実用になる水準に届いたので、まず出しました。発展途中のこの不完全なモデルを、皆さんもご自身の手で、ご自身なりの完成に近づけてみませんか。私も作業を続け、成果が出たら更新版を出していきます。皆さんの版も Hugging Face や GitHub に出していただけたら嬉しいです。

## 得られるもの

測定したシングルストリーム設定――一人のユーザー・一本の
ストリーム・C=1・thinking オフ――では、サービングしたモデルは
二台の DGX Spark 上で約 35 tok/s で応答します。正確な値は
35.09 tok/s で、pair 1・NCCL/IB (RDMA)・MTP ドラフト K=2 の
構成を、64 本の日本語散文プロンプト・`max_tokens=512`・
`temperature=0` の固定ルーラーで測定しました。ストックの参照行
――pair 2・sockets・ドラフト無し・同じルーラーの先頭
8 プロンプト――は 10.75 tok/s です。最も近い行は下の結果表に
あります(pair 1・sockets・K=2 の stock + MTP は expert parallel
オフで 19.05 tok/s、同じ route h + MTP K=2 構成の pair 1・sockets
は expert parallel オンで 24.01 tok/s――`ep` 列を参照)。

## 速度の由来

| 要素 | 内容 | 実測 |
|---|---|---|
| ベース速度 | route h 再量子化(BF16 dense 側を W4A16 NVFP4 化) | 18.98 tok/s、ドラフト無し――pair 1・sockets・64 プロンプト全件(ドラフト無しの RDMA 行は未測定。stock のドラフト無し行は pair 2・sockets・先頭 8 プロンプトで 10.75 tok/s) |
| 投機デコード | チェックポイント自身の MTP ヘッド、K=2 | 受理率 0.62、周期あたり約 2.24 出力トークン |
| 合成 | 再量子化 + MTP K=2 + NCCL over RDMA | **35.09 tok/s** ―― pair 1・RDMA・64 プロンプト全件・K=2 |

貪欲デコードでは、投機ステップはベースモデルが選んだはずのトークン
だけを受理するので、出力分布は変わりません。したがって品質の差分は
再量子化だけに帰属します。それを測るのが次の節です。

## コスト

再量子化は速度と引き換えに、上限のある品質差分を受け入れます。
ゲートは速度測定の前に宣言したもので、リリース構成はすべての
基準を通過する必要がありました。

| 基準 | 閾値 | stock | route h | 結果 |
|---|---|---|---|---|
| 64 プロンプト・ルーラー上の退化出力 | 0 | 0 | 0 | PASS |
| ホールドアウトした passage の perplexity 比 | <= 1.10 | 1.0 | 1.051 | PASS |
| eval-200 実効精度(150 項目、0/50 の tool フロアを除く) | stock より 0.02 以上低くない | 0.447 | 0.453 (+0.0067) | PASS |
| 約 2000 トークン probe の TTFT | stock の 1.2x 以内 | 2.380 s | 2.436 s (x1.02) | PASS |

perplexity 比 1.051 は stock 比 5.1% の増加で、宣言したゲートの内側
ではありますが、同等ではなく増加です。eval-200 は 200 項目の素点で
+0.005、ゲートが採点する 150 の実効項目で +0.0067 動きました。
この規模のセットではこれほど小さな差を分解できず、それ単体で
同等性を証明するものではありません。

両スコアと各指標が実用上意味するものを含む完全な差分を
`results/quality.tsv` からレンダリングしたもの:

<!-- quality:start -->
| metric | stock | route h | what it means for a user |
|---|---|---|---|
| Perplexity, held-out passages | 11.94 | 12.54 (ratio 1.051 = +5.1%) | How surprised the model is by fresh text; +5.1% is a real but small regression inside the declared 1.10 gate. |
| eval-200: reason | 32/50 | 27/50 | Multi-step reasoning items solved; -5 of 50 on a 50-item column -- too few items to separate a real regression from noise. |
| eval-200: trap | 13/50 | 16/50 | Trick-question resistance; +3 of 50 on a 50-item column -- same limit applies. |
| eval-200: tool | 0/50 | 0/50 | Tool-call items score zero on both checkpoints -- the prompts never name a callable tool, so the column is 0 by construction and cannot judge either side. |
| eval-200: longread | 22/50 | 25/50 | Long-document comprehension; +3 of 50 on a 50-item column -- same limit applies. |
| eval-200: total | 67/200 | 68/200 (+0.005 acc) | Overall accuracy moved +0.005 raw (+0.0067 on the 150 live items the gate scores, tool floor excluded), inside the declared gate; by itself it does not prove equivalence. |
| TTFT, ~2000-token probe | 2.380 s | 2.436 s (x1.02) | Delay before the first token on a long prompt; a 2% increase measured on this probe -- no user-perception test was run. |
| Degenerate outputs, 64-prompt ruler | 0 | 0 | Empty or looping completions; zero on both sides. |
| 13-item evaluate suite | -- | pending | The end-to-end serve evaluation (short/long decode, parallel-4, agentic tool-use, long-context, trick questions); queued on this configuration -- this row fills in when it lands. |
<!-- quality:end -->

## 再現方法

完全なランブック(前提条件・正確なコマンド・予想所要時間・ディスク
必要量)は [AGENTS.md](AGENTS.md) にあります。要点だけ言うと、
`requant/requant.py --target h` でチェックポイントを再量子化し、
RDMA イメージとオーバーレイをビルドし、`serve/start-head.sh` と
`serve/start-worker.sh` でサービングし、`requant/verify.py check`
でゲートを通し、`bench/measure.py` で測定します。

## うまくいかなかったものと時期

同じルーラーで測定したものです。探索空間を記録に残すために並べます。

<!-- failures:start -->

| item | what | cause | date |
|---|---|---|---|
| **moe-dsl-kernel-overlay** | Three successive failures. First, the patched kernel path was dead code: the backend is gated to device family 100 and never instantiates on sm_121, so an apparent -4 ms TPOT delta was run-to-run noise. Second, after re-hooking the class the deployment selects, every call failed a static eligibility check (the model's SwiGLU clamp limit) and silently fell back to stock: 256 calls, zero dispatches. Third, a host-side cute.make_layout call raised a TypeError at first use. | Instrumented, then fixed or abandoned each time; the overlay is not in the released config. | 2026-09-15 |
| **fp8-dense** | FP8-quantized dense linear variants ran slower than W4A16 NVFP4 on this stack. | Measured slower; not adopted. | 2026-09-15 |
| **mtp-k-ge-3** | K=3 gave 19.39 tok/s and K=4 gave 17.44 tok/s vs K=2's 19.83 on an earlier requant route (sockets, pair 1); on stock, K=4 gave 15.2 and K=5 gave 13.1. | Per-position acceptance drops more quickly than the extra draft tokens save steps. K=2 was the best of the K values measured on this route; on route h + RDMA a first sweep-A pass measured K=1 at 34.28 and K=3 at 31.98 tok/s, and the full same-checkpoint sweep in both K orders is still pending. | 2026-09-16 |
| **ep-off** | Removing --enable-expert-parallel gained ~0.7%. | Inside the pair-to-pair offset; not a lever. | 2026-09-15 |
| **eager-mode** | --enforce-eager is ~10 ms/step faster on the stock single-stream step, but the gain mostly does not carry into MTP decode. | The step it speeds up is not the step speculation runs. Not adopted. | 2026-09-15 |
| **k4-k5-boot-oom** | K=4/K=5 failed to boot for the route-h checkpoint at these memory settings (128 GB unified memory); the stock checkpoint booted both. One K=4 attempt was killed externally by an orphaned pipeline's timeout cleanup; the genuine cause: the Ray memory monitor's default 95% threshold OOM-killed the worker right after speculator CUDA-graph capture, and at K=5 the capture took 103 s and 3.03 GiB, leaving 0.89 GiB of KV cache where 0.92 GiB was required. | Memory accounting and capture cost scale with K; the fix direction is a smaller drafter, not bigger K. | 2026-09-15 |
| **requant-loader-write** | A fused-KDA write missing the NVFP4 global scale produced KeyError ...weight_scale_2 at load. | Writer bug; now covered by verify.py config and the kda-quant overlay. | 2026-09-15 |
| **nccl-ll-symm-ar** | NCCL low-latency envs and symmetric-memory all-reduce booted but returned empty response bodies or died under concurrency. | Broken output path; unsafe, not adopted. | 2026-09-15 |
| **ev-ordering** | The expensive drafter-training data chain (extraction, two generation passes, conversion; ~5 h of wall time on the 2-GPU pair, ~10 GPU-hours) was queued ahead of the cheap K sweep; the trained drafter's holdout top-1 came out ~0.08, and the sweep showed K=2 was already the best of the measured values. | Sequenced by pipeline momentum instead of expected information per GPU-hour. | 2026-09-15 |

<!-- failures:end -->

上記および下記の測定の時系列です。

| 日付 | 節目 |
|---|---|
| 2026-09-13 | pair 2 での stock ベースライン: C=1 で 10.75 tok/s |
| 2026-09-15 | stock + 公式 MTP K=2: pair 1 で 19.05 tok/s、pair 2 で 17.84 tok/s。代替レバーの多くはこの期間に失敗 |
| 2026-09-16 | route h 再量子化、RDMA イメージ、pair 1 でヘッドライン 35.09 tok/s |

## NVIDIA NVFP4 チェックポイントを出発点にした理由

公式の `nvidia/GLM-5.3-Flash-NVFP4` リリースは NVIDIA 自身のツール
チェーンで量子化されており、そのモデルカードは実質的に精度劣化の
無いことを示す BF16 対 NVFP4 のベンチマーク数値を公開しています
(モデルカードのリビジョン `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`)。
GPQA Diamond 0.9217 -> 0.9211、SciCode 0.5621 -> 0.5769、MMMU Pro
0.7688 -> 0.763、AA-LCR 0.71 -> 0.7106、IFBench 0.613 -> 0.6054、
Terminal Bench 2.1 は 0.8258 -> 0.8315 です。これが信頼できる土台に
なる理由です。私の route h がやるのは、公式リリースが BF16 のまま
残した層へ同じ 4-bit 処理を広げることだけで、トレードオフが見え
続けるようゲートの数値を速度の数値の隣に置いています。

## Stage 1 — リリース構成

- **route `h` 再量子化** — attention linear(KDA fused `in_proj`/`out`
  射影、MLA q/kv/o、indexer `wq_b`)、shared-expert の
  `gate`/`up`/`down`、そして `lm_head` を W4A16 NVFP4 へ変換。
  router・norm・embedding は BF16 のままです。
- **NCCL over RDMA** — 派生イメージ(`docker/Dockerfile.nccl-ib`)が
  公式イメージの上に Ubuntu questing の `rdma-core` を入れ、NCCL
  NET/IB プラグインが init できるようにします。`serve/nccl-ib.sh`
  がノードごとの HCA と RoCE v2 GID を導出し、`uverbs*`/`rdma_cm`
  デバイスと `IPC_LOCK`/無制限 `memlock` を渡します。
- **公式 MTP ドラフト K=2** — モデル自身の MTP 層に対して
  `num_speculative_tokens=2`。`mtp-bf16` オーバーレイがドラフトの
  サブツリーを非量子化で組み立てることで有効になります(ストック
  イメージはドラフトにターゲットの NVFP4 quant config を継承させ、
  BF16 の MTP 重みのロードでクラッシュします)。

## 速度結果

固定ルーラー: 64 本の日本語散文プロンプト、`temperature=0`、
`max_tokens=512`、空の assistant 継続で thinking をスキップ。
特記が無ければ C=1 が 64 プロンプトすべてを逐次ストリームし、
プロンプトごとの TTFT/TPOT を取ります。同梱の `serve/` スクリプトは
`--max-model-len 16384`、`--gpu-memory-utilization 0.85`、FP8 KV
キャッシュを固定しますが、測定行はバリアントで走っています――
2026-09-13 の stock ベースラインは `--max-model-len 131072`、
c1pair と route h の行は `--gpu-memory-utilization 0.86`、
expert parallel は `ep` = `off` と記した stock + MTP の二行(17.84 と 19.05 tok/s)でオフ、それ以外ではオンでした。各行の正確なフラグは `results/results.tsv` の `source_log` が指す `results/logs/` 以下の同梱ファイルにあります。

数え方:

- **tok/s** — パス全体の completion token 合計 / 壁時計時間合計。
  TTFT を含む、窓全体のスループットです。
- **TPOT med** — プロンプトごとの `(wall - TTFT) / (tokens - 1)` の
  中央値。
- **weighted TPOT** — プロンプト全体での
  `sum(wall - TTFT) / sum(tokens - 1)`。トークン加重の平均です。
  TPOT med の逆数は tok/s とは別の統計で、一致は期待しません
  (同じ実行で 51.5 ms -> 19.42 対 実測 18.98)。
- **accept** — パス全体の spec-decode カウンタ差分:
  accepted / draft tokens。ヘッドライン行の周期あたり平均受理長
  (出力数)は 2.244 で、1.244 の受理ドラフトトークンにボーナス
  トークンを足したものです。
- **quality gate / TTFT gate** — 上記の四条件ゲートを行ごとに
  二列へ分けた表示です。degenerate・PPL・eval-200 の各脚は
  チェックポイントに紐付きます(PASS (PPL x1.051) のセル)。TTFT 脚は
  transport 依存で、route h の sockets サービング上で一度だけ
  probe しました。sockets 行は "verified on sockets"、RDMA 行は
  "verified on sockets; not re-run under RDMA" と表示し、TTFT probe を
  通していない transport では四条件 PASS を主張しません。
  transport ごとの TTFT は各行の TTFT med 列を読んでください。

<!-- results:start -->

| checkpoint | pair | transport | K | ep | tok/s | TPOT med ms | weighted TPOT ms | TTFT med s | accept | quality gate | TTFT gate |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| stock | 2 | sockets | - | on | 10.75 † | 95.3 † | 92.5 † | 0.361 † | - | ref | — |
| stock + MTP | 2 | sockets | 2 | on | 17.71 | 55.5 | 55.7 | 0.447 | 0.6128 | ref | — |
| stock + MTP | 2 | sockets | 2 | off | 17.84 | 55.2 | n/m | 0.441 | 0.6183 | ref | — |
| stock + MTP | 1 | sockets | 2 | off | 19.05 | 51.9 | 51.7 | 0.430 | 0.6175 | ref | — |
| stock | 2 | NCCL/IB (RDMA) | - | on | 14.32 † | 69.3 † | 69.3 † | 0.349 † | - | ref | — |
| stock + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 23.93 | 41.0 | 41.2 | 0.366 | 0.6137 | ref | — |
| route h | 1 | sockets | - | on | 18.98 | 51.5 | 52.1 | 0.330 | - | PASS (PPL x1.051) | verified on sockets |
| route h + MTP | 1 | sockets | 2 | on | 24.01 | 40.7 | 41.0 | 0.389 | 0.6221 | PASS (PPL x1.051) | verified on sockets |
| **route h + MTP** | **1** | **NCCL/IB (RDMA)** | **2** | **on** | **35.09** | **27.9** | **27.9** | **0.320** | **0.6221** | **PASS (PPL x1.051)** | **verified on sockets; not re-run under RDMA** |
| route g + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 29.46 | 33.3 | 33.3 | 0.340 | 0.6182 | PASS (PPL x0.999) | verified on sockets; not re-run under RDMA |
| route h | 1 | NCCL/IB (RDMA) | 0/1/3/4 | on | pending |  |  |  |  |  |  |
| route h + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 34.47 | 28.2 | 28.4 | 0.313 | 0.6221 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | NCCL/IB (RDMA) | 2 | on | 34.48 | 28.3 | 28.4 | 0.317 | 0.6193 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 23.75 | 41.2 | 41.4 | 0.394 | 0.6213 | PASS (PPL x1.051) | verified on sockets |
| route h + MTP | 1 | NCCL/IB (RDMA) | 2 | on | 35.12 | 27.9 | 27.9 | 0.313 | 0.6193 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 24.35 | 40.4 | 40.4 | 0.39 | 0.6174 | PASS (PPL x1.051) | verified on sockets |
| route c + MTP | 1 | sockets | 2 | on | 19.83 | 49.6 | n/m | 0.435 | 0.6218 | PASS | verified on sockets |
| route c + MTP | 1 | sockets | 3 | on | 19.39 | 50.9 | n/m | 0.461 | 0.4925 | PASS | verified on sockets |
| route c + MTP | 1 | sockets | 4 | on | 17.44 | 56.7 | n/m | 0.446 | 0.3925 | PASS | verified on sockets |

<!-- results:end -->

† これらの行はルーラー全 64 本ではなく先頭 8 プロンプトで C=1 を
回したものです。

繰り返し測定(いずれも 2026-09-16 に着地):

- **二つ目のペア** — 同じヘッドライン構成をもう一組のノードペアで
  34.47 tok/s。同一構成の繰り返し実行で見ているおおよそ 5-7% の
  ペア間オフセットの内側です。
- **公開用の再測定** — 重複の無い新規 64 プロンプト集合で
  prefix キャッシュ無効・暖機別計上の TCP/RDMA 交互再実行:
  RDMA で 34.48 と 35.12 tok/s、TCP で 23.75 と 24.35 tok/s。
  RDMA 側の再測定は元の 35.09 の ~2% 以内に収まりました。
- pass ごとの GPU テレメトリ(両ノードで 2 s 間隔の nvidia-smi;
  SM clock median / power median–max / temp max): 34.48 の pass は
  2190 MHz、24.9–27.1 W、68 °C。23.75 の pass は 2190 MHz、
  20.5–23.6 W、64 °C。35.12 の pass は 2190 MHz、24.5–26.7 W、
  65 °C。24.35 の pass は 2190 MHz、20.6–24.3 W、65 °C。
  二つ目のペアの再実行はテレメトリ未採取(n/m)です。

未測定の行:

- **K sweep** — 同じチェックポイントで route h + RDMA の
  K=0/1/3/4 全掃引。暖機・順序効果を見るため両方向の K 順で
  キュー済み。sweep A の一パスは既に着地しており――pair 1・
  RDMA で K=1 が 34.28 tok/s、K=3 が 31.98――、全掃引の検収が
  終わるまで K=2 は測定した値の中での最良であり、証明された
  最適値ではありません。

ヘッドライン実行(pair 1、2026-09-16): 923.43 s の壁時計で
32407 completion tokens、失敗リクエストはゼロ、finish reason は
61 `length` / 3 `stop`。stock チェックポイントの C=32 集計は
94.66 tok/s(sockets)と 109.03 tok/s(RDMA)で、route h の C=32 行は
まだありません。

## 名前について

侘びは、不完全なものや素朴なものを受け入れ、その中に豊かさを見出す日本の感覚です。このリリースは作りかけのものをそのまま公開するもので、改良は実測できたものだけを取り込みます。

## Stage 2

より速い構成が後続することがあります。

## ファイル

```
AGENTS.md                   reproduction runbook (prereqs, commands, times)
CONTRIBUTORS.md             who worked on this
requant/requant.py          weight-only repack, targets a-e/g/h
requant/verify.py           config verifier + gate v2 (capture/check)
requant/build-mtp-draft.py  BF16 MTP draft dir from the checkpoint
overlays/                   image-source patchers (fail-closed anchors):
  patch-kda.py              restore quant_config in fused KDA members
  patch-mla.py              pass quant_config into Glm5NextMLAAttention
  patch-mtp.py              build the MTP draft subtree unquantized
  apply-step-attr-patch.py + step_attr.py   CUDA-event step attribution
  fetch-image-file.sh       snapshot a file out of the image
  build-overlays.sh         fetch + patch everything into _build/
docker/Dockerfile.nccl-ib   derived image: questing rdma-core on noble
serve/                      start-head.sh / start-worker.sh / nccl-ib.sh
bench/                      measure.py + prompts-64.jsonl + eval-200.jsonl
tests/                      run-tests.sh (offline smoke) +
                            check-md-invariants.py (wording-only edit check)
```

`eval-200.jsonl` のプロンプト二件は公開用に引用符を打ち直しました
(ASCII 安全な引用 / 指示文の言い換え)。期待回答と採点は変わって
いません。

## Contributors

- tenhkspark(メンテナ) — 心配して見守り、出すと言った人
  (匿名ハンドルのみ、実名は非公開)。
- Claude Fable 5.1 (Anthropic) — 方向付け、実験設計、レビューと検収。
- Claude Opus 5 (Anthropic) — リリース前レビュー。
- Astra (OpenAI, via pi) — 測定と計画への敵対的レビュー。
- Devin SWE-2 (Cognition) — 実装、診断、キューのステージング、
  リポジトリのドラフト作成。
- GLM-5.3 (Z.ai) — 実装担当。
- GLM-5.3-Flash (Z.ai) — 実装担当と要約。

## License

Apache-2.0。LICENSE を参照。モデルの重みはこのリポジトリに含みま
せん。`requant/requant.py` で NVIDIA チェックポイントを各自で再量子
化してください。
