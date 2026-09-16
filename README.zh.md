# glm53-flash-nvfp4-2node

最新发布: **v1**(git 标签 `v1`)。重新量化后的检查点发布在
Hugging Face 上,名为 `tenhkspark/GLM-5.3-Flash-NVFP4-Wabi`——
一个仓库、git 标签 `v1` 与 `v2`,最新版本始终以本文件开头为准。

我在两台 NVIDIA DGX Spark 节点上用 Ray 张量并行(TP=2)服务
`GLM-5.3-Flash-NVFP4` 的方案,以及用来恢复单流解码速度的仅权重
重新量化。这里所有内容都是我自己的实现和在自己硬件上的实测——
*enjoying the incomplete*。

不包含模型权重。输入检查点是 NVIDIA 的 `GLM-5.3-Flash-NVFP4`
发布版;重新量化脚本会改写该检查点中的 BF16 dense linear,而不是
重新分发任何内容。请把脚本指向你自己的副本。

还有好几个改进的想法留在清单上,但它已经达到了我觉得可用的水准,所以先发布出来。这个仍在发展中的不完美模型,也请您亲手把它推向您自己心目中的完成形态。我也会继续改进,一有实测成果就发布更新版。如果您的版本也能发布到 Hugging Face 或 GitHub,我会很高兴。

## 得到什么

在实测的单流设定——一个用户、一条流、C=1、thinking 关闭——下,
服务出的模型在两台 DGX Spark 上以约 35 tok/s 作答。精确数字是
35.09 tok/s,是在 pair 1、NCCL/IB (RDMA)、MTP 草稿 K=2 的配置下,
用 64 条日语散文提示、`max_tokens=512`、`temperature=0` 的固定
标尺测得的。stock 的参照行——pair 2、sockets、无草稿、同一标尺的
前 8 条提示——为 10.75 tok/s。最接近的各行见下方结果表(pair 1、
sockets、K=2 的 stock + MTP 在 expert parallel 关闭下为
19.05 tok/s,同一 route h + MTP K=2 配置的 pair 1、sockets 行在
expert parallel 开启下为 24.01 tok/s——见 `ep` 列)。

## 速度从何而来

| 要素 | 内容 | 实测 |
|---|---|---|
| 基础速度 | route h 重新量化(BF16 dense 侧改为 W4A16 NVFP4) | 18.98 tok/s,无草稿——pair 1、sockets、全部 64 条提示(无草稿的 RDMA 行待测;stock 的无草稿行为 pair 2、sockets、前 8 条提示,10.75 tok/s) |
| 投机解码 | 检查点自带的 MTP 头,K=2 | 接受率 0.62,每周期约 2.24 个输出 token |
| 合成 | 重新量化 + MTP K=2 + NCCL over RDMA | **35.09 tok/s** —— pair 1、RDMA、全部 64 条提示、K=2 |

在贪心解码下,投机步骤只接受基础模型本会选择的 token,因此不改变
输出分布。所以质量差异只归属于重新量化——下一节衡量的正是它。

## 代价

重新量化用有界的质量差换取速度。门限在任何速度测量之前就已声明;
发布配置必须通过全部标准。

| 标准 | 阈值 | stock | route h | 结果 |
|---|---|---|---|---|
| 64 条提示标尺上的退化输出 | 0 | 0 | 0 | PASS |
| 留出 passage 的 perplexity 比 | <= 1.10 | 1.0 | 1.051 | PASS |
| eval-200 有效准确率(150 项,剔除 0/50 的 tool 地板) | 不低于 stock 超过 0.02 | 0.447 | 0.453 (+0.0067) | PASS |
| 约 2000 token probe 的 TTFT | stock 的 1.2x 以内 | 2.380 s | 2.436 s (x1.02) | PASS |

perplexity 比 1.051 是相比 stock 增加 5.1%——在声明的门限之内,但
是增加而非持平。eval-200 在 200 项原始分上移动了 +0.005,在门限
实际评分的 150 个有效项上移动了 +0.0067。这种规模的集合无法分辨
这么小的差异,本身也不能证明等价。

包含两边得分与每项指标对用户含义的完整差异,由
`results/quality.tsv` 渲染:

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

## 如何复现

完整的操作手册——前提条件、精确命令、预期耗时和磁盘需求——在
[AGENTS.md](AGENTS.md) 中。简言之:用 `requant/requant.py --target h`
重新量化检查点,构建 RDMA 镜像和 overlay,用 `serve/start-head.sh` 与
`serve/start-worker.sh` 起服务,用 `requant/verify.py check` 过门限,
再用 `bench/measure.py` 测量。

## 哪些没有成功,以及时间

均在同一把标尺上测过;列出它们是为了把搜索空间留在记录里。

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

上述与下文测量值的时间线:

| 日期 | 节点 |
|---|---|
| 2026-09-13 | pair 2 上的 stock 基线:C=1 下 10.75 tok/s |
| 2026-09-15 | stock + 官方 MTP K=2:pair 1 上 19.05 tok/s,pair 2 上 17.84 tok/s;多数替代杠杆在此窗口内失败 |
| 2026-09-16 | route h 重新量化、RDMA 镜像、pair 1 上头条数字 35.09 tok/s |

## 为什么从 NVIDIA NVFP4 检查点出发

官方 `nvidia/GLM-5.3-Flash-NVFP4` 发布版使用 NVIDIA 自己的工具链
量化,其模型卡公布了显示基本没有精度损失的 BF16 对 NVFP4 基准数字
(模型卡修订 `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`):
GPQA Diamond 0.9217 -> 0.9211,SciCode 0.5621 -> 0.5769,MMMU Pro
0.7688 -> 0.763,AA-LCR 0.71 -> 0.7106,IFBench 0.613 -> 0.6054,
Terminal Bench 2.1 为 0.8258 -> 0.8315。这使它成为可信赖的基座。
我的 route h 只是把同样的 4-bit 处理扩展到官方发布留在 BF16 的层,
并把门限数字放在速度数字旁边,让权衡始终可见。

## Stage 1 — 发布配置

- **route `h` 重新量化** — attention linear(KDA fused `in_proj`/`out`
  投影、MLA q/kv/o、indexer `wq_b`)、shared-expert 的
  `gate`/`up`/`down` 以及 `lm_head` 转换为 W4A16 NVFP4。router、
  norm 和 embedding 保持 BF16。
- **NCCL over RDMA** — 派生镜像(`docker/Dockerfile.nccl-ib`)在官方
  镜像上安装 Ubuntu questing 的 `rdma-core`,使 NCCL NET/IB 插件
  能够 init;`serve/nccl-ib.sh` 按节点推导 HCA 与 RoCE v2 GID,并
  传入 `uverbs*`/`rdma_cm` 设备以及 `IPC_LOCK`/不限制 `memlock`。
- **官方 MTP 草稿 K=2** — 对模型自身的 MTP 层使用
  `num_speculative_tokens=2`,由 `mtp-bf16` overlay 以非量化方式
  构建草稿子树来启用(stock 镜像会让草稿继承目标的 NVFP4 quant
  config,在加载 BF16 MTP 权重时崩溃)。

## 速度结果

固定标尺:64 条日语散文提示,`temperature=0`,`max_tokens=512`,
通过空的 assistant 续写跳过 thinking。除非注明,C=1 顺序流式跑完全部
64 条提示并记录每条提示的 TTFT/TPOT。随附的 `serve/` 脚本固定
`--max-model-len 16384`、`--gpu-memory-utilization 0.85`、FP8 KV
缓存;但实测各行跑的是变体——2026-09-13 的 stock 基线用
`--max-model-len 131072`,c1pair 与 route h 各行用
`--gpu-memory-utilization 0.86`,expert parallel 在标为 `ep` = `off` 的两条 stock + MTP 行(17.84、19.05 tok/s)上关闭,其余各行开启。各行的确切标志见 `results/results.tsv` 的 `source_log` 所指向的 `results/logs/` 下的随附文件。

计数方式:

- **tok/s** — 整轮 completion token 总数 / 总 wall time,含 TTFT。
  整窗吞吐量。
- **TPOT med** — 各条提示 `(wall - TTFT) / (tokens - 1)` 的中位数。
- **weighted TPOT** — 全部提示的
  `sum(wall - TTFT) / sum(tokens - 1)`;按 token 加权的均值。
  TPOT med 的倒数与 tok/s 是不同的统计量,不期望一致
  (同一轮中 51.5 ms -> 19.42 对实测 18.98)。
- **accept** — 整轮的 spec-decode 计数器差分:
  accepted / draft tokens。头条行每周期的平均接受长度(输出数)为
  2.244,即 1.244 个接受草稿 token 加奖励 token。
- **quality gate / TTFT gate** — 上述四条件门限按行拆为两列显示。
  degenerate/PPL/eval-200 三项挂在检查点上(PASS (PPL x1.051)
  单元格);TTFT 项依赖于 transport,只在 route h 的 sockets 服务上
  探测过一次。sockets 行显示 "verified on sockets",RDMA 行显示
  "verified on sockets; not re-run under RDMA",不会在未经 TTFT 探测的
  transport 上宣称四条件 PASS。各 transport 的 TTFT 以每行的
  TTFT med 列为准。

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

† 这些行只用标尺前 8 条提示跑了 C=1,不是全部 64 条。

重复测量(均于 2026-09-16 落地):

- **第二个 pair** — 同一头条配置在另一组节点对上读出
  34.47 tok/s,落在相同配置重复运行约 5-7% 的 pair 间偏移之内。
- **发布用复测** — 在不重叠的新 64 条提示集上以 TCP/RDMA 交替
  复跑头条配置,禁用 prefix 缓存、暖机单独计:RDMA 下
  34.48 和 35.12 tok/s,TCP 下 23.75 和 24.35 tok/s。RDMA 侧
  复测与原 35.09 相差 ~2% 以内。
- 每个 pass 的 GPU 遥测(两台节点均以 2 s 间隔采集 nvidia-smi;
  SM clock median / power median–max / temp max):34.48 pass 为
  2190 MHz、24.9–27.1 W、68 °C;23.75 pass 为 2190 MHz、
  20.5–23.6 W、64 °C;35.12 pass 为 2190 MHz、24.5–26.7 W、
  65 °C;24.35 pass 为 2190 MHz、20.6–24.3 W、65 °C。第二个
  pair 的复跑未采集遥测(n/m)。

待测的行:

- **K sweep** — 同一检查点上 route h + RDMA 的 K=0/1/3/4 全量扫掠,
  为观察暖机/顺序效应以两种 K 顺序排队。sweep A 已有一轮落地——
  pair 1、RDMA 下 K=1 为 34.28 tok/s、K=3 为 31.98——但在全量扫掠
  验收完成前,K=2 是我测过的值中最好的——而不是已证实的最优。

头条运行(pair 1,2026-09-16):923.43 s 的 wall time 内
32407 个 completion tokens,零失败请求,finish reason 为
61 `length` / 3 `stop`。stock 检查点的 C=32 聚合为
94.66 tok/s(sockets)和 109.03 tok/s(RDMA);route h 还没有 C=32
行。

## 关于名字

侘び(Wabi)是日本的一种审美意识:接纳不完美或朴素的事物,并从中发现丰富之美。本次发布的是尚在完善中的作品,按现状公开;改进只在经过实测后才会纳入。

## Stage 2

后续可能出现更快的配置。

## 文件

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

`eval-200.jsonl` 中有两条提示为发布重写了引用形式(ASCII 安全引用
/ 指示语改写);期望答案与评分不变。

## Contributors

- tenhkspark(维护者) — 那个一直担心、观望,最后说"发吧"的人
  (仅以匿名 handle 署名,不公开真名)。
- Claude Fable 5.1 (Anthropic) — 方向、实验设计、评审与验收。
- Claude Opus 5 (Anthropic) — 发布前评审。
- Astra (OpenAI, via pi) — 对测量与计划的对抗性评审。
- Devin SWE-2 (Cognition) — 实现、诊断、队列排布、本仓库草稿撰写。
- GLM-5.3 (Z.ai) — 实现担当。
- GLM-5.3-Flash (Z.ai) — 实现担当与摘要。

## License

Apache-2.0,见 LICENSE。模型权重不属于本仓库;请用
`requant/requant.py` 自行对 NVIDIA 检查点做重新量化。
