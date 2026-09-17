# glm53-flash-nvfp4-2node

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh.md) · [한국어](README.ko.md)

两台 NVIDIA DGX Spark 节点服务 `GLM-5.3-Flash-NVFP4`,单流解码
达到 **35.09 tok/s**——一个用户、一条流、C=1、thinking 关闭——
在 node pair 1 上经 NCCL/IB (RDMA) 测得,使用检查点自带的 MTP
draft、K=2,标尺固定为 64 条日语散文提示,`max_tokens=512`,
`temperature=0`。实际使用时,这意味着日语文字大致以我在屏幕上
阅读的速度到达,而不是一阵一阵地冒出来。

本仓库就是这个数字背后的配方:跨两台节点的 Ray 张量并行
(TP=2)、我用来恢复单流解码速度的仅权重重新量化(route h)、
一个支持 RDMA 的派生镜像,以及让 MTP draft 能够加载的 overlay。
这里的一切都是我自己的实现、在我自己的硬件上的实测——
*enjoying the incomplete*。

下面每个数字都带着它的条件(pair、transport、提示条数、K、
expert parallel、`max_tokens`),速度表中的每一行都注明了它出自
哪份日志。让比较完全同条件的那一行 stock 已于 2026-09-17 落地——
14.39 tok/s,pair 1、经 RDMA、无 draft、全部 64 条提示——所以重新
量化与 draft 现在各自都只跨越单一条件变化就能分开。

这份配方也已经从外部倒着跑过一遍。在另一组节点对上从空的工作目录
出发,手上只有已发布的 [AGENTS.md](AGENTS.md) 和
`scripts/agent-run.sh`,这次运行搭起了一套可服务的配置,并在自己的
10 条提示冒烟测试上测得 37.08 tok/s——相对 35.09 的头条数字是 +5.7%,
落在随附的检查器允许的 7% 带内。下面的"从已发布的操作手册复现"一节
说明这项测试证明了什么、没有证明什么。

不包含模型权重。输入检查点是 NVIDIA 的 `GLM-5.3-Flash-NVFP4`
发布版;重新量化脚本改写的是该检查点中的 BF16 dense linear,
而不是重新分发任何内容。请把脚本指向你自己的副本。

清单上还有几个改进的想法,但这个模型已经达到我认为可用的水准,
所以我现在就发布。它仍在进行中:请把它拿去,朝你自己心目中的
完成形态推进。我会继续做下去,有结果就发布更新;也很高兴在
Hugging Face 和 GitHub 上看到你的版本。

发布标签:**v1**。重新量化后的检查点以
`tenhkspark/GLM-5.3-Flash-NVFP4-Wabi` 发布在 Hugging Face 上,
一个仓库,同一个标签。目前还没有 v2:只有当别的配置通过同一道
门限,并且在同一把标尺上测得更快时,v2 标签才会出现。

## 为什么用两台节点

检查点在磁盘上约 204 GB(route h 改写后约 190 GB),放不进单台
节点的 128 GB 统一内存,所以模型以 TP=2 切分到两台节点上,每一个
解码步都要跨过它们之间的链路。两台节点通过各自 ConnectX-7 级别
的端口,用一根 200GbE QSFP 铜缆端到端直连,路径中没有交换机;
这里所有数字都是在这组直连的 pair 上测得的。同一个 netdev 承载
两种 transport:`NCCL_IB=1` 让 NCCL 跑在 RoCE v2 上(表中的
"NCCL/IB (RDMA)"),`NCCL_IB=0` 退回 TCP("sockets")。

由于这条链路位于解码循环之内,transport 在这里不是细节——它是
下面三个杠杆之一。

## 发布配置

- **route `h` 重新量化** — attention linear(KDA fused
  `in_proj`/`out` 投影、MLA q/kv/o、indexer `wq_b`)、shared-expert
  的 `gate`/`up`/`down` 以及 `lm_head` 转换为 W4A16 NVFP4。router、
  norm 和 embedding 保持 BF16。
- **NCCL over RDMA** — 派生镜像(`docker/Dockerfile.nccl-ib`)在
  官方镜像上安装 Ubuntu questing 的 `rdma-core`,使 NCCL NET/IB
  插件能够 init;`serve/nccl-ib.sh` 按节点推导 HCA 与 RoCE v2 GID,
  并传入 `uverbs*`/`rdma_cm` 设备以及 `IPC_LOCK`/不限制 `memlock`。
- **官方 MTP draft、K=2** — 对模型自身的 MTP 层使用
  `num_speculative_tokens=2`,由 `mtp-bf16` overlay 以非量化方式
  构建 draft 子树来启用(stock 镜像会让 draft 继承目标的 NVFP4
  quant config,在加载 BF16 MTP 权重时崩溃)。

## 每项改动带来什么

诚实的分解需要同一组 pair、同一种 transport、同一把标尺上的四行。
四行现在都已测得:

| 步骤 | 它隔离出什么 | 行 | 实测 |
|---|---|---|---|
| stock, no draft, RDMA | 快速 transport 上的起点 | pair 1, 64 prompts | 14.39 tok/s |
| route h, no draft, RDMA | 只有重新量化 | pair 1, 64 prompts | 27.15 tok/s |
| route h + MTP K=2, RDMA | 在重新量化之上加 draft | pair 1, 64 prompts | 35.09 tok/s |
| route h + MTP K=2, sockets | 同一检查点与 draft,换成慢速 transport | pair 1, 64 prompts | 24.01 tok/s |

上面那三行现在共享全部条件——pair 1、NCCL/IB (RDMA)、同样的 64 条
日语散文提示、`temperature=0`、`max_tokens=512`、expert parallel
开启、thinking 关闭——所以重新量化与 draft 都可以干净地与其他一切分开:

- stock、无 draft:14.39 tok/s,TPOT 中位数 69.0 ms,TTFT 中位数
  0.293 s,零失败请求。
- route h、无 draft:27.15 tok/s,TPOT 中位数 35.7 ms,加权 TPOT
  36.4 ms,TTFT 中位数 0.269 s,零失败请求。
- route h + MTP K=2:35.09 tok/s,TPOT 中位数 27.9 ms,TTFT 中位数
  0.320 s,接受率 0.6221,零失败请求。

27.15 / 14.39 = 1.89、35.09 / 27.15 = 1.29,所以在这个配置上,重新
量化大约值 1.89x,draft 在其之上大约值 1.29x。这是本 README 中仅有的
两个只跨越单一条件变化取得的比值;其余任何两行之间的差异都不止一处,
所以我不去相除。两个比值的分子分母都是单轮测量,相同配置的轮间离散
约为 2%。

今天我能说的,每一条都限定在同一组 pair、同一种 transport 之内:

- **draft。** route h、pair 1、64 条提示:经 RDMA 时无 draft 为
  27.15 tok/s、MTP K=2 时为 35.09(x1.29);经 sockets 时为 18.98
  与 24.01,四者都开启 expert parallel。K=2 各轮的接受率为
  0.6174–0.6221,即每个 draft 周期约 2.24 个输出 token。
- **transport。** route h + MTP K=2、pair 1、64 条提示:sockets 下
  24.01 tok/s,RDMA 下 35.09;两次发布用复跑在同一组 pair 上读出
  sockets 23.75 / 24.35、RDMA 34.48 / 35.12,所以 transport 的
  差距可以复现。
- **重新量化。** stock 对 route h、pair 1、64 条提示,两边都经
  RDMA、无 draft、开启 expert parallel:stock 为 14.39 tok/s,
  route h 为 27.15(x1.89)。更早的那个 stock 参照——在 pair 2 上、
  经 sockets、用标尺的前 8 条提示、以 `--max-model-len 131072` 测得的
  10.75 tok/s——不能当这个分母:用它去除会一次混入三处条件变化,
  所以我不去相除。

## 代价

重新量化用质量换速度。这笔取舍只在下面四条标准上有界——全部是
日语、单轮、thinking 关闭——在它们之外没有任何界限。门限在任何速度
测量之前就已声明;发布配置必须通过全部四条标准:

| 标准 | 阈值 | stock | route h | 结果 |
|---|---|---|---|---|
| 64 条提示标尺上的退化输出 | 0 | 0 | 0 | PASS |
| 8 句探针上的 perplexity 比 | <= 1.10 | 1.0 | 1.051 | PASS |
| eval-200 有效准确率(150 项,剔除 0/50 的 tool 地板) | 比 stock 低不超过 0.02 | 0.447 | 0.453 (+0.0067) | PASS |
| 约 2000 token 探针上的 TTFT | stock 的 1.2x 以内 | 2.380 s | 2.436 s (x1.02) | PASS |

**perplexity 是怎么测的。** perplexity 在为这项测试写的八个日语
短句(共 249 个字符)上测量,它们内联定义在 `requant/verify.py`
中;这些句子不取自任何公开语料,我也不主张做过训练集排除。每个
句子都作为只做 prefill 的请求发给服务端(`/v1/completions`,
`echo=true`、`max_tokens=1`、`prompt_logprobs=1`、temperature 0);
把八个句子中实际提示 token 的对数概率累加起来,对这一条汇总的
token 流取平均负对数似然,再取指数,得到一个数字。stock 与候选
用同一个函数、对同一个已服务的模型测量,门限是候选/stock 的比值
<= 1.10——发布的 route h 得到 12.54,stock 为 11.94,比值 1.051。
样本很小,所以这只检查是否有严重退化,而不是检查持平。

**eval-200 是怎么评分的。** eval-200 是随本仓库提供的一组固定的
200 条日语单轮提示(`bench/eval-200.jsonl`):reason、trap、tool、
longread 各 50 条,难度混合 easy/medium/hard。每一项都在服务端以
贪心方式作答(temperature 0、`max_tokens` 384、通过空的 assistant
续写跳过 thinking),并由 `requant/verify.py` 中的确定性规则评分:
当该项的评分标准要求只给出答案时用忽略空白的精确匹配,否则用
子串包含;trap 项必须带有拒答且不含数字,tool 项必须按顺序点出
所需的函数以及预期的最终值。tool 列在任何检查点上都是 0/50,
这是结构性的——提示中从不点出可调用的函数,也没有发送任何 tool
schema——所以门限评的是其余 150 个有效项,要求候选与 stock 的
差距在 0.02 以内且请求零错误(stock 0.447 -> route h 0.453)。
原始总分 67/200 -> 68/200 只是为了透明而报告;它们包含那个恒为
零的列,不是通过/不通过的判据。各类别的变动并不独立:trap 的评分器
奖励拒答,所以一个更爱含糊其辞的模型会在那里得分更高,而在 reason
上更低。总分净 +1 也可能是一次有方向的退化造成的。

**TTFT 探针**是把同样那八个句子连接十二次,作为一条提示发送;
门限把候选在它上面的 time-to-first-token 与 stock 比较,是在
route h 的 sockets 服务上测的。

perplexity 比 1.051 是相比 stock 增加 5.1%——在声明的门限之内,
但是增加,而非持平。eval-200 在 200 项原始分上移动了 +0.005,在门限
实际评分的 150 个有效项上移动了 +0.0067。150 项的集合在 95% 置信度
下只能把准确率分辨到约 +-0.08(配对,假设约 20% 的项会翻转),所以
我测到的 +0.0067 与零无法区分——真实退化 0.05 也同样无法区分。0.02
的门限阈值比这个集合能分辨的粒度更细:一个真的比 stock 差 0.06-0.08
的检查点,多半仍然能通过它。把这道门限读成"没有退化"是错的;它只
排除了大的退化。

在贪心采样下,投机解码的设计是只接受目标模型本会产生的 token,
所以原则上 draft 不会改变输出分布,质量差只归属于重新量化。
我没有测量这一等价性,所以请把它当作设计上的论证,而不是实测
结果。

完整的差异——两边的得分,以及每个指标在使用中意味着什么——由
`results/quality.tsv` 渲染:

<!-- quality:start -->
| metric | stock | route h | what it means for a user |
|---|---|---|---|
| Perplexity, 8-sentence Japanese probe | 11.94 | 12.54 (ratio 1.051 = +5.1%) | How surprised the model is by the probe text; +5.1% is a real but small regression inside the declared 1.10 gate. The probe is 249 characters of Japanese written for this test -- nothing was held out from training, and a sample this small checks for gross degradation, not for parity. |
| eval-200: reason | 32/50 | 27/50 | Multi-step reasoning items solved; -5 of 50 on a 50-item column -- too few items to separate a real regression from noise. |
| eval-200: trap | 13/50 | 16/50 | Trick-question resistance, graded mechanically: an answer passes if it contains a refusal marker and no digits. That rule rewards hedging, so a checkpoint that became more evasive would gain here while losing on reason -- which is the direction this pair of columns actually moved (+3 trap, -5 reason). I did not test whether the two moves share that cause; do not read the +1 net total as 'no change'. |
| eval-200: tool | 0/50 | 0/50 | Tool-call items score zero on both checkpoints -- the prompts never name a callable tool, so the column is 0 by construction and cannot judge either side. |
| eval-200: longread | 22/50 | 25/50 | Long-document comprehension; +3 of 50 on a 50-item column -- same limit applies. |
| eval-200: total | 67/200 | 68/200 (+0.005 acc) | Overall accuracy moved +0.005 raw (+0.0067 on the 150 live items the gate scores, tool floor excluded), inside the declared gate; by itself it does not prove equivalence. The 95% interval on this difference is about +-0.08, which is four times wider than the 0.02 gate threshold. |
| TTFT, ~2000-token probe | 2.380 s | 2.436 s (x1.02) | Delay before the first token on a long prompt. The 2% gap is the median of three runs of the same prompt and is the same size as the run-to-run spread I measure on identical configurations, so this probe shows no TTFT regression it could have detected -- it does not show that TTFT is unchanged. No user-perception test was run. |
| Degenerate outputs, 64-prompt ruler | 0 | 0 | Empty or looping completions; zero on both sides. Zero out of 64 is consistent with a true rate of up to about 5% (rule of three), and the detector only catches empty output, repeated-token runs and exactly periodic loops -- it cannot see a fluent answer that is wrong, truncated or off-topic. |
| 13-item evaluate suite | -- | pending | The end-to-end serve evaluation (short/long decode, parallel-4, agentic tool-use, long-context, trick questions); queued on this configuration -- this row fills in when it lands. |
<!-- quality:end -->

上面每一行都是日语、单轮、thinking 关闭。唯一能覆盖长上下文、并发
请求和 agentic 工具调用的是那套还在 pending 的 13 项套件;在它落地
之前,这些维度在这个检查点上都未经测试。

**这道门限没有测什么。** 上面每个数字都是日语、单轮、thinking
关闭、贪心、上下文不到 2k token、一次一条请求。我没有测过这个检查点
在英语或任何其他语言上的表现,没有测过代码正确性、多轮对话、工具
调用(tool 列是结构性的零)、指令遵循、长上下文——出货的服务窗口是
262144 token,但上面每一项探针都只有 2k 左右——
安全行为,也没有测过开启 thinking 的情况——而这正是这个
模型家族通常的用法。重新量化改写的是 dense linear 和 lm_head,所以
那里恰恰是退化可能躲过这四项探针的地方。如果你依赖其中任何一项,
请在采用这个检查点之前自己测;requant/verify.py 只要一个 flag 就能
换成另一组提示。

### route g,如果你想要更小的质量差

`requant/requant.py --target g` 写出一个更保守的重新量化:在
pair 2 上经 RDMA、MTP K=2、同一把 64 条提示的标尺,route g 测得
29.46 tok/s,perplexity 比为 0.999;在同一组 pair、transport、K
与标尺下,route h 为 34.47 tok/s、比值 1.051。我发布 h,是因为
门限的职责是给质量差设上界,而不是把它最小化,而 1.051 在我测量
之前声明的上界之内。如果你更愿意用速度去换更小的质量差,g 只是
改一个 flag。

## 如何复现

完整的操作手册——前提条件、精确命令、预期耗时和磁盘需求——在
[AGENTS.md](AGENTS.md) 中。简言之:用 `requant/requant.py --target h`
重新量化检查点,构建 RDMA 镜像和 overlay,用 `serve/start-head.sh`
与 `serve/start-worker.sh` 起服务,用 `requant/verify.py check`
过门限,再用 `bench/measure.py` 测量。提示集的说明见
[bench/README.md](bench/README.md)。

## 从已发布的操作手册复现

上面每个数字都是写配方的人在写配方的那台机器上测的——这是任何速度
主张里最弱的一环。所以我把这份配方从外部倒着跑了一遍:在另一组节点
对上用一个空的工作目录,仓库骨架就是已发布的样子,指示只有
[AGENTS.md](AGENTS.md) 和 `scripts/agent-run.sh`,操作机就用那组 pair
自己的一台节点。这次运行构建了 RDMA 镜像与 overlay,为它自己的两台
节点填好 `setup.env`,起了 head 与 worker,过了配置门限并完成测量——
没有从我的工作树里拿任何东西。它跳过了一步:重新量化本身。那组
pair 上已经放着发布用的 route-h 检查点,顶替了下载,所以这项测试
复现的是从检查点到测得 token 的全部过程,而不是权重的改写。

- 在这次运行自己的 10 条提示冒烟测试上为 37.08 tok/s,TPOT 中位数
  26.5 ms,接受率 0.6177,零失败请求。
- 相对 35.09 tok/s 的发布行是 +5.7%,落在 `scripts/verify-result.py`
  为 pair 间偏移允许的 7% 带内。
- TTFT 中位数 0.36 s,而发布行是 0.32 s——慢 12.5%,超出那条带。
  检查器对 TTFT 只报告而不设门限,而这一行正是它仍然被报告的理由。

这项测试没有证明的是:冒烟那一轮是 10 条提示,发布行是 64 条,而且
提示集不同,所以 +5.7% 不是同条件的比较——一轮 10 条提示的轮间离散
比标尺大得多。它证明的是:已发布的步骤,单独从一个空目录照着做,
能够到达与我发布的那套相邻的服务配置。走到那一步需要对操作手册和
驱动脚本做七处修正,列在下面的"哪些没有成功,以及时间"里;每一处
都是靠这次干净运行的失败发现的,从我自己的树里一处都看不见。

## 速度结果

固定标尺:64 条日语散文提示,`temperature=0`,`max_tokens=512`,
通过空的 assistant 续写跳过 thinking。除非注明,C=1 顺序流式跑完
全部 64 条提示,并记录每条提示的 TTFT/TPOT。随附的 `serve/` 脚本
固定 `--max-model-len 262144`、`--max-num-seqs 2`、
`--gpu-memory-utilization 0.88`、FP8 KV 缓存。下面实测各行跑的不是
这套,而是测量用的设置——绝大多数是 `--max-model-len 16384` 配
`--max-num-seqs 20`,2026-09-13 的 stock 基线用
`--max-model-len 131072` 配 32 个槽位,pair 2 的 stock RDMA 那一行
用 16384 配 32 个槽位;c1pair 与 route h 各行用
`--gpu-memory-utilization 0.86`,expert parallel 在标为 `ep` =
`off` 的两条 stock + MTP 行(17.84 和 19.05 tok/s)上关闭,其余
各行开启。各行的确切 flag 见 `results/results.tsv` 中该行
`source_log` 条目所指向的 `results/logs/` 下的文件。

**为什么出货的窗口是 262144,而表里不是。** 16384 是测量用的值——
我最早把 MTP 拉起来的那个长度——它就这样一路留在上面每一次比较里。
2026-09-17 重测的结果是:发布的配置在
`--max-model-len 262144` `--max-num-seqs 2`
`--gpu-memory-utilization 0.88` 下,对同一把 64 条提示的标尺读出
35.15 tok/s,对比 16384 / 20 / 0.86 的 35.09——窗口拉长 16 倍,
在约 2% 的轮间离散里看不出差别。长窗口的代价是并发而不是速度——
随附脚本只要 2 个槽位而不是 20 个,是因为在 307200 下引擎用 20 个
槽位起不来,而我没有在 262144 下重测这个上限;0.89 的利用率在这对
节点上有过被拒绝启动的记录。窗口装得下,原因之一正是
重新量化:同样在 0.88 下,官方检查点的上限是 156672 token(vLLM 在
拒绝启动时会把这个上限打印出来),而 route h 在 262144 下能起来。
我没有在接近这个长度的提示上测过任何东西——见上面"这道门限没有测
什么"。

我的计数方式:

- **tok/s** — 整轮 completion token 总数 / 整轮 wall time,含
  TTFT。整窗吞吐量。
- **TPOT med** — 各条提示 `(wall - TTFT) / (tokens - 1)` 的
  中位数。
- **weighted TPOT** — 全部提示的
  `sum(wall - TTFT) / sum(tokens - 1)`;按 token 加权的均值。
  五份出货的 JSON(四份发布复跑加上 route h 无 draft RDMA 那一行)
  把它作为显式记录的字段带上;另外九行,我自己用同一公式,从这个
  仓库之外、清洗之前的 factory 日志里每一轮的逐条记录算出来,写进了
  对应行出货的 `results/logs/` 条目里,并在那里注明了推导方式。
  2026-09-13 基线在任何地方都找不到逐条记录,它自己的数字(92.5 ms)
  是各提示上的普通平均——是另一种统计量。所以这一行,以及我还没有
  算出来的其余各行,一律写 `n/m`,而不是把三种定义混进同一列。TPOT med 的
  倒数与 tok/s 同样是不同的统计量,不应期望二者一致
  (51.5 ms 按算术是 19.42 tok/s,而那一轮实测为 18.98)。
- **accept** — 整轮的 spec-decode 计数器差分:accepted / draft
  tokens。头条行每周期的平均接受长度(每周期的输出数)为 2.244:
  1.244 个被接受的 draft token 加上奖励 token。
- **quality gate / TTFT gate** — 上述四条件门限,按行拆成两列。
  degenerate / PPL / eval-200 三项挂在检查点上("PASS (PPL
  x1.051)" 单元格)。TTFT 项依赖于 transport,只在 route h 的
  sockets 服务上探测过一次:sockets 行写 "verified on sockets",
  RDMA 行写 "verified on sockets; not re-run under RDMA"——没有
  任何一行在 TTFT 探针从未见过的 transport 上宣称四条件 PASS。
  各 transport 的数字以每行自己的 TTFT med 列为准。

<!-- results:start -->

| checkpoint | pair | transport | K | ep | tok/s | TPOT med ms | weighted TPOT ms | TTFT med s | accept | quality gate | TTFT gate |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| stock | 2 | sockets | - | on | 10.75 † | 95.3 † | n/m | 0.361 † | - | ref | — |
| stock + MTP | 2 | sockets | 2 | on | 17.71 | 55.5 | 55.73 | 0.447 | 0.6128 | ref | — |
| stock + MTP | 2 | sockets | 2 | off | 17.84 | 55.2 | n/m | 0.441 | 0.6183 | ref | — |
| stock + MTP | 1 | sockets | 2 | off | 19.05 | 51.9 | 51.77 | 0.430 | 0.6175 | ref | — |
| stock | 2 | NCCL/IB (RDMA) | - | on | 14.32 † | 69.3 † | 69.29 | 0.349 † | - | ref | — |
| stock | 1 | NCCL/IB (RDMA) | - | on | 14.39 | 69.0 | n/m | 0.293 | - | ref | — |
| stock + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 23.93 | 41.0 | 41.17 | 0.366 | 0.6137 | ref | — |
| route h | 1 | sockets | - | on | 18.98 | 51.5 | 52.12 | 0.330 | - | PASS (PPL x1.051) | verified on sockets |
| route h | 1 | NCCL/IB (RDMA) | - | on | 27.15 | 35.7 | 36.4 | 0.269 | - | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 24.01 | 40.7 | 40.96 | 0.389 | 0.6221 | PASS (PPL x1.051) | verified on sockets |
| **route h + MTP** | **1** | **NCCL/IB (RDMA)** | **2** | **on** | **35.09** | **27.9** | **27.92** | **0.320** | **0.6221** | **PASS (PPL x1.051)** | **verified on sockets; not re-run under RDMA** |
| route g + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 29.46 | 33.3 | 33.35 | 0.340 | 0.6182 | PASS (PPL x0.999) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | NCCL/IB (RDMA) | 1 | on | 34.28 | 28.6 | n/m | 0.304 | 0.7979 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | NCCL/IB (RDMA) | 3 | on | 31.98 | 30.8 | n/m | 0.331 | 0.483 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 34.47 | 28.2 | 28.43 | 0.313 | 0.6221 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | NCCL/IB (RDMA) | 2 | on | 34.48 | 28.3 | 28.4 | 0.317 | 0.6193 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 23.75 | 41.2 | 41.4 | 0.394 | 0.6213 | PASS (PPL x1.051) | verified on sockets |
| route h + MTP | 1 | NCCL/IB (RDMA) | 2 | on | 35.12 | 27.9 | 27.9 | 0.313 | 0.6193 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 24.35 | 40.4 | 40.4 | 0.39 | 0.6174 | PASS (PPL x1.051) | verified on sockets |
| route c + MTP | 1 | sockets | 2 | on | 19.83 | 49.6 | n/m | 0.435 | 0.6218 | PASS | verified on sockets |
| route c + MTP | 1 | sockets | 3 | on | 19.39 | 50.9 | n/m | 0.461 | 0.4925 | PASS | verified on sockets |
| route c + MTP | 1 | sockets | 4 | on | 17.44 | 56.7 | n/m | 0.446 | 0.3925 | PASS | verified on sockets |

<!-- results:end -->

† 这些行我只用标尺前 8 条提示跑了 C=1,不是全部 64 条。

分解中 route h 无 draft 的那一条已于 2026-09-16 落地(27.15 tok/s),
stock 无 draft 的那一条已于 2026-09-17 落地(14.39 tok/s),两者都在
pair 1 上经 RDMA、用同一把 64 条提示的标尺测,所以给重新量化算加速
倍数时所需要的分母已经测到:27.15 / 14.39 = 1.89。

这张表里故意没有 "vs stock" 列:有了这一列就会想给每一行都算比值,
而这里大多数 stock 行取自不同的 pair、不同的 transport 或 8 条而非
64 条提示,那样的比值会把三处条件变化折进一个数字里。只差一处条件的
那两对,写在上面这一段和"每项改动带来什么"里。

K 也还没有定论。在 pair 1 上经 RDMA、用 64 条提示的标尺,K=1 测得
34.28 tok/s,K=2 在三轮中测得 34.48 / 35.09 / 35.12;相同配置的
轮间离散约为 2%,所以这个实验无法把 K=1 与 K=2 区分开。K=3 在同一
组 pair 与 transport 上测得 31.98,接受率从 K=2 的 0.6193 降到
K=3 的 0.483。我发布 K=2,是因为它是我测过的值中最好的,而不是
因为 K=1 被排除了。

重复测量(均于 2026-09-16 落地):

- **第二组 pair** — 同一头条配置在另一组节点对上读出 34.47 tok/s,
  落在我在相同配置的重复运行中看到的约 5-7% 的 pair 间偏移之内。
- **发布用复测** — 在全新、不重叠的 64 条提示集上,以 TCP/RDMA
  交替方式复跑头条配置,禁用 prefix 缓存,暖机单独计:RDMA 下
  34.48 和 35.12 tok/s,TCP 下 23.75 和 24.35 tok/s。RDMA 的复跑
  与原来的 35.09 相差在 ~2% 以内。
- 每一轮的 GPU 遥测,在两台节点上以每 2 s 一次用 `nvidia-smi`
  采样,覆盖该轮全程(每台节点 523–772 个样本)。四轮的所有样本
  中 SM clock 都是 2190 MHz。功耗在四轮中都落在 20–27 W 区间,GPU
  最高温度为 68 °C;每一轮具体落在这个区间的哪个位置,我没有留下
  逐轮的细分记录。这些逐样本的 CSV 不属于本仓库;`results/logs/`
  下那四份复跑 JSON 属于本仓库。

头条运行(pair 1,2026-09-16)在 923.43 s 的 wall time 内产出
32407 个 completion token,零失败请求;出货的那份日志只记录了总计,
没有记录每条请求的 finish reason,但我从清洗之前的 factory 日志里
数了一遍,头条本身那 64 条提示的构成同样是 61 `length` / 3 `stop`。
四次发布复跑
直接记录了这一项,构成完全一样——也就是说,
多数提示是撞上 512 token 的上限,而不是自己停下来。在 stock
检查点上,C=32 的聚合值为 sockets 下 95.08 tok/s(pair 2,64 条
提示)和 RDMA 下 109.03 tok/s(pair 2,64 条提示);这两轮都是用
32 个槽位跑的,出货的 `--max-num-seqs 2` 复现不了,所以请把它们当作
测量用的数字,而不是发布脚本会给出的数字。route h 目前
还没有 C=32 的行。

### 按提示类型

64 条提示的标尺是散文。为了看清文本的类型会把数字挪动多少,我在发布
配置上另跑了几组 32 条提示的集合:pair 1、NCCL/IB (RDMA)、route h +
MTP K=2、`temperature=0`、expert parallel 开启,每一组都各自以 C=1
跑一轮。这些集合不是那把 64 条提示的标尺,也不是它的子集——提示不同,
条数是 32 而不是 64——所以它们不进上面的表。

- 散文,`max_tokens=512`:34.87 tok/s,TPOT 中位数 28.1 ms,TTFT
  中位数 0.319 s。
- 散文,`max_tokens=128`:32.80 tok/s,TPOT 中位数 28.9 ms,TTFT
  中位数 0.293 s。
- 代码,`max_tokens=512`:38.20 tok/s,TPOT 中位数 25.6 ms,TTFT
  中位数 0.342 s。
- 结构化输出与 JSON:这两类也已测过,读数见下面一段(不含 TTFT 中位数)。

结构化提示在 512 token 下为 34.84 tok/s、128 token 下为 33.51(TPOT 中位数分别是 27.4 和 27.6 ms);JSON 形态的提示为 35.35 与 33.30(25.0 和 25.2 ms);代码在 128 token 下为 36.08(25.1 ms)。在这个配置上,四种类型都落在 32.8 到 38.2 tok/s 之间,每一轮都没有失败的提示。

散文这一读数 34.87 tok/s,与同一服务配置下 64 条提示标尺的 35.09
贴在一起——这正是我希望在两组不同的散文集合之间看到的一致。更早
一轮跑同一组 32 条散文提示、同样的配置,读到的是 28.88 tok/s:那
一轮落在一个慢窗口里,加权 TPOT 为 34.0 ms,而头条那一轮是 27.9 ms,
同时接受率没有动(0.6189 对 0.6221),所以变的是每个解码周期的时间,
不是 draft。两个读数都留在记录里。它们之间的离散,是这台机器上被
别的事情碰到时,一轮 32 条提示能够产生的幅度,比 64 条提示的标尺在
复跑之间显示的约 2% 要宽。

## 哪些没有成功,以及时间

都在同一把标尺上测过;列出它们是为了把搜索空间留在记录里。

<!-- failures:start -->

| item | what | cause | date |
|---|---|---|---|
| **moe-dsl-kernel-overlay** | Three successive failures. First, the patched kernel path was dead code: the backend is gated to device family 100 and never instantiates on sm_121, so an apparent -4 ms TPOT delta was run-to-run noise. Second, after re-hooking the class the deployment selects, every call failed a static eligibility check (the model's SwiGLU clamp limit) and silently fell back to stock: 256 calls, zero dispatches. Third, a host-side cute.make_layout call raised a TypeError at first use. | Instrumented, then fixed or abandoned each time; the overlay is not in the released config. | 2026-09-15 |
| **fp8-dense** | FP8-quantized dense linear variants ran slower than W4A16 NVFP4 on this stack (pair 1, sockets, 64-prompt ruler): no draft 9.3 tok/s / TPOT 107.1 ms against W4A16's 18.98 / 51.5; with the MTP K=2 draft 16.49 tok/s / TPOT 59.7 ms against W4A16's 24.01 / 40.7. | Measured slower; not adopted. | 2026-09-15 |
| **mtp-k-ge-3** | K=3 gave 19.39 tok/s and K=4 gave 17.44 tok/s vs K=2's 19.83 on an earlier requant route (sockets, pair 1); on stock (pair 2, sockets, expert parallel on), K=4 gave 15.2 and K=5 gave 13.1. | Per-position acceptance drops more quickly than the extra draft tokens save steps. K=2 was the best of the K values measured on this route; on route h + RDMA a first sweep-A pass measured K=1 at 34.28 and K=3 at 31.98 tok/s; the reverse-order pass was run and failed to boot at every K including K=0 (only 99.88 GiB was free on the device at startup against the 104.65 GiB the utilization target asks for), which is a startup-memory limit rather than a K effect, so the full same-checkpoint sweep in both K orders has still not produced a complete set of numbers. | 2026-09-16 |
| **ep-off** | Removing --enable-expert-parallel gained ~0.7% (pair 2, sockets, MTP K=2, 64-prompt ruler: 17.71 -> 17.84 tok/s). | Inside the pair-to-pair offset; not a lever. | 2026-09-15 |
| **eager-mode** | --enforce-eager speeds up the stock single-stream step by 7.0 ms on the 256-step attribution totals (86.95 vs 93.98 ms) and by 13.4 ms on the C=1 TPOT median (77.5 vs 90.9 ms), but the gain mostly does not carry into MTP decode. | The step it speeds up is not the step speculation runs. Not adopted. | 2026-09-15 |
| **k4-k5-boot-oom** | On the STOCK checkpoint at these memory settings (128 GB unified memory per node, max seq len 16384, gpu-memory-utilization 0.86) the engine refused to start at K=5 twice: after the speculator's CUDA-graph capture (103 s and 3.03 GiB on one rank) the KV cache left was smaller than one max-seq-len request needs (0.89 GiB available against 0.92 GiB needed). One K=4 attempt died at the same capture step. Restricting the CUDA-graph capture sizes to the two sizes the C=1 measurement needs made both boot (READY 879 s and 901 s) and produced the K=4 15.2 and K=5 13.1 rows above, so this is a capture-size configuration limit, not a hard K ceiling. On route h + RDMA, K=1, K=3 and K=4 all booted (READY 885 s, 906 s and 918 s); the K=4 arm was stopped by the operator before it measured, so route h has no K=4 number, and K=5 was never run on route h at all. | Speculator graph capture grows with K and is taken out of the KV budget: supported for stock K=5 by the capture size and the KV figures above, and worked around by capturing fewer sizes. Two unrelated events are kept out of this column: an earlier K=4 attempt was killed by an orphaned pipeline's cleanup, and a later reverse-order route-h sweep failed to boot at every K including K=0 because only 99.88 GiB was free on the device at startup against the 104.65 GiB the utilization target asks for. | 2026-09-15 |
| **requant-loader-write** | A fused-KDA write missing the NVFP4 global scale produced KeyError ...weight_scale_2 at load. | Writer bug; now covered by verify.py config and the kda-quant overlay. | 2026-09-15 |
| **nccl-ll-symm-ar** | NCCL low-latency envs and symmetric-memory all-reduce booted but returned empty response bodies or died under concurrency. | Broken output path; unsafe, not adopted. | 2026-09-15 |
| **ev-ordering** | The expensive drafter-training data chain (extraction, two generation passes, conversion; ~5 h of wall time on the 2-GPU pair, ~10 GPU-hours) was queued ahead of the cheap K sweep; the trained drafter's holdout top-1 came out ~0.08, and the sweep showed K=2 was already the best of the measured values. | Sequenced by pipeline momentum instead of expected information per GPU-hour. | 2026-09-15 |

<!-- failures:end -->

另有七处失败来自上面那次干净运行,全部出在操作手册和驱动脚本里,
而不是模型、transport 或检查点——这类问题只有在从空目录照着字面
执行指示时才会露面。其中五处:一个 `mktemp` 模板被操作手册自己的
shell 拒绝(结尾的 `X` 少于三个);一份驱动脚本写了却从未 source 的
serve 配置;镜像内的 helper 以 root 身份创建的文件,落进了下一步
无法写入的目录;操作机与服务 head 是同一台机器的情形,脚本却按
SSH 到自己来处理;以及把 JSON 参数两侧引号吃掉的 shell 引用。每一处
都在这里发布的文件中修好了,而不是在那次运行里。

上述与下文测量值的时间线:

| 日期 | 节点 |
|---|---|
| 2026-09-13 | pair 2 上的 stock 基线:C=1 下 10.75 tok/s |
| 2026-09-15 | stock 上的官方 MTP K=2:pair 1 上 19.05 tok/s,pair 2 上 17.84;多数替代杠杆在这一窗口内失败 |
| 2026-09-16 | route h 重新量化、RDMA 镜像、pair 1 上头条数字 35.09 tok/s |

## 为什么我从 NVIDIA NVFP4 检查点出发

官方 `nvidia/GLM-5.3-Flash-NVFP4` 发布版是用 NVIDIA 自己的工具链
量化的,它的模型卡公布了 BF16 对 NVFP4 的基准数字,显示基本没有
精度损失(模型卡修订
`09b04e5e74bca08ca8549fc736d4cdd8624bfde3`):GPQA Diamond
0.9217 -> 0.9211,SciCode 0.5621 -> 0.5769,MMMU Pro
0.7688 -> 0.7630,AA-LCR 0.7100 -> 0.7106,IFBench 0.6130 -> 0.6054,
Terminal-Bench 2.1 0.8258 -> 0.8315。这使它成为可信赖的基座。
我的 route h 只是把同样的 4-bit 处理扩展到官方发布留在 BF16 的
那些层,并把门限数字放在速度数字旁边,让这项权衡始终可见。模型卡
上的那些数字来自标准的公开基准,与上文的 eval-200 总分不可比较,
后者是由本仓库自己的严格评分器打出来的。

## v2 还剩下什么

这是一份我知道还没做完的清单,不是路线图。我不预测其中任何一项值
多少;我测到数字之后才会发布数字。

**速度,仍未关闭**

- 我自己的 NVFP4 MoE kernel。它从未真正在一次服务请求中跑起来:
  三次尝试分别挂到了 GB10 构建不会实例化的类、挂错了类,以及一个
  抛出异常并悄悄把 overlay 关掉的主机侧调用。在引擎日志里出现证明
  它还活着的那一行之前,没有什么可测的。
- all-reduce。在我做过的那一次每步归因里(eager、stock、单流、经
  sockets),跨节点等待对端占了约 87 ms 解码步中的 20.8 ms。我没有在
  RDMA 下重跑过同样的归因——试过一次,但在启动阶段就挂了,没有产出
  任何步级数据——所以这段等待在快 transport 上还剩多少,我并不知道。
  RDMA 那些行 tok/s 更高,是整轮的实测,不是对这一项的归因。减少
  集合通信本身的次数,我也没有试过。
- 一个训练出来的 draft 模型。检查点自带的 MTP 头接受它自己提出的
  token 的 0.62。用这个模型自己的输出训练出的 draft 可能接受得更多,
  那会缩短每一个周期。我的第一次尝试数据不足——594 条训练记录、约
  301k token,而配方目标约为 3M token,holdout 每槽命中率
  0.03-0.08——在值得测量之前大约还需要十倍的数据。
- 投机深度。K=1(34.28)与 K=2(35.09)在这个实验的轮间离散下分
  不开,而 K=3 更慢。更好的 draft 会改变那个最优点的位置。

**质量,尚未测量**

- 英语和其他语言。这里每一项探针都是日语。
- 代码正确性。代码提示出现在速度表里,在质量表里一处也没有。
- 多轮对话、工具调用、指令遵循、长上下文——出货的窗口是 262144
  token,但我没有在接近这个长度的提示上测过任何东西——安全行为,
  以及开启 thinking 的模型;而这正是这个家族通常的用法。
- 更大语料上的 perplexity,并带上置信区间。目前的探针是八个句子,
  代码也不保留逐 token 的值,所以这个比值没有区间可附。
- eval-200 上的配对检验。评分器丢弃了逐项的结果,所以那 150 个项目
  无法按它们本来的样子成对检验。

**复现,尚未完成**

- 干净运行跳过了下载和重新量化本身:权重已经在节点上了。从空盘
  开始、经过约 204 GB 下载和约 30 分钟重新量化的完整路径,还没有
  任何人端到端跑过。
- 那次运行用的是 10 条提示,而不是参考数字所出自的 64 条。

**我已经知道行不通的。** dense linear 用 FP8、关掉 expert
parallelism、在散文上把 K 取到 3 或更大,以及强制 eager 模式。这些
数字和它们的测量条件都在上面的失败表里;重试其中任何一项之前,先
去看那里。

## 没有尽头的那部分

把模型做快花了一天的实验。把结果做到可以发布花的时间更长,而那正是
我没有准备好的那部分。

我加的每一层检查都查出了东西,而每一处发现都是真的。一道检查用词和
泄露的机器检查第一次就通过了,于是我又加了一道把正文里每个数字追回
日志的检查——其中十四个追不回来。我修好那些,再请一位没有上下文的
读者通读全文,结果头条的比较原来是把两个在不同硬件、不同 transport、
不同提示条数下测得的数字相除。我修好那个,再把随附的证据与表格
对照,发现一个吞吐数字取自它自己日志里错误的那一行。我修好那个,
再把量化脚本与正文的说法逐条核对,发现一个 helper 在你把它指向错
的检查点时会悄悄构建出一个坏掉的 draft。我修好那个,再请人用对抗
的眼光读质量主张,于是学到:四个类别中有一个的评分器会奖励模型含糊
其辞——而那正是我的数字移动的方向。

这些都不是粗心。每一处都需要另一种看法。而模式始终一样:加一道
检查,发现一件事,修好它,而修补都很小。花代价的不是检查本身;花
代价的是意识到永远还有下一层。

到某个时刻,诚实的做法是停下,不是因为工作完成了,而是因为下一层
的价值已经低于把东西发出去。我停在这里。我知道没有测的事情都列在
上面它们自己的那一节里,而我没有想到要检查的事情,正是这是 v1 而
不是终版的理由。如果你发现了一件,那说明这套机制在起作用,我宁愿
听到它,也不愿听不到。

## 关于名字

侘び(Wabi)是日本的一种感受:接纳不完美或朴素的事物,并从中
发现丰富之美;本次发布是一件仍在进行中的作品,我按现状公开它,
改进则在测量之后纳入。

## 接下来做什么

先把按类型的其余几组跑完,然后是两种 K
顺序下的完整 K sweep;如果某个配置在这把标尺上超过 35.09 tok/s 并
通过同一道门限,它就会作为 v2 发布。

## 文件

```
AGENTS.md                   reproduction runbook (prereqs, commands, times)
CONTRIBUTORS.md             who worked on this
LICENSE / NOTICE            Apache-2.0 and attribution
setup.env.example           per-node paths and image tags (copy to setup.env)
requant/requant.py          weight-only NVFP4 quantization, targets a-e/g/h
requant/verify.py           config verifier + gate v2 (capture/check)
requant/build-mtp-draft.py  BF16 MTP draft dir from the STOCK checkpoint
overlays/                   image-source patchers (fail-closed anchors):
  patch-kda.py              restore quant_config in fused KDA members
  patch-mla.py              pass quant_config into Glm5NextMLAAttention
  patch-mtp.py              build the MTP draft subtree unquantized
  apply-step-attr-patch.py + step_attr.py   CUDA-event step attribution
  fetch-image-file.sh       snapshot a file out of the image
  build-overlays.sh         fetch + patch everything into _build/
docker/Dockerfile.nccl-ib   derived image: questing rdma-core on noble
serve/                      start-head.sh / start-worker.sh / nccl-ib.sh
                            + serve.env.example
scripts/agent-run.sh        one-shot driver for the whole runbook
scripts/verify-result.py    checks a measured row against the expected one
bench/                      measure.py + prompts-64.jsonl + eval-200.jsonl
                            + README.md (what the sets are)
results/                    results.tsv, quality.tsv, failures.tsv and the
                            sanitised per-run logs the tables cite
tests/                      run-tests.sh (offline smoke) +
                            check-md-invariants.py (wording-only edit check)
```

有两条 `eval-200.jsonl` 提示为发布重写了引用形式(ASCII 安全的
引号 / 指示语改写);期望答案与评分不变。`bench/prompts-64.jsonl`
与实测时完全一致——见 [bench/README.md](bench/README.md)。

## 谁做了什么

本仓库中的设计决定、验收标准和每一项测量都出自我。实现是与下面
列出的、也记在 [CONTRIBUTORS.md](CONTRIBUTORS.md) 中的 AI 席位
一起完成的;门限、各轮运行和数字在发布前都由我核对过。

## Contributors

- tenhkspark(维护者) — 那个一直担心、观望,最后说"发吧"的人。
- Claude Fable 5.1 (Anthropic) — 方向、实验设计、评审与验收。
- Claude Opus 5 (Anthropic) — 发布前评审。
- Astra (OpenAI) — 对测量与计划的对抗性评审。
- Devin SWE-2 (Cognition) — 实现、诊断、运行实验队列、本仓库草稿
  撰写。
- GLM-5.3 (Z.ai) — 实现席位。
- GLM-5.3-Flash (Z.ai) — 实现席位与摘要。

## License

Apache-2.0,见 LICENSE。模型权重不属于本仓库;请用
`requant/requant.py` 自行对 NVIDIA 检查点做重新量化。
