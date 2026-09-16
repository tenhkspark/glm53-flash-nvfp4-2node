# glm53-flash-nvfp4-2node

최신 릴리스: **v1**(git 태그 `v1`). 재양자화한 체크포인트는
Hugging Face에서 `tenhkspark/GLM-5.3-Flash-NVFP4-Wabi`로 공개되어
있다——하나의 리포지토리에 git 태그 `v1`과 `v2`가 있고, 최신 버전은
항상 이 파일의 맨 위가 가리킨다.

두 대의 NVIDIA DGX Spark 노드에서 Ray 텐서 병렬화(TP=2)로
`GLM-5.3-Flash-NVFP4`를 서빙하는 나의 레시피와, 싱글 스트림 디코드
속도를 회복하기 위해 사용하는 가중치 전용 재양자화. 여기 있는 모든
것은 나 자신의 구현과 내 하드웨어에서의 직접 측정값이다——
*enjoying the incomplete*.

모델 가중치는 포함하지 않는다. 입력 체크포인트는 NVIDIA의
`GLM-5.3-Flash-NVFP4` 릴리스이며, 재양자화 스크립트는 무엇인가를
재배포하는 대신 그 체크포인트의 BF16 dense linear를 다시 쓴다.
스크립트는 각자의 사본을 가리키면 된다.

아직 개선 아이디어는 여럿 남아 있지만, 실용으로 쓸 만한 수준에 도달했기에 우선 공개합니다. 발전 도중인 이 불완전한 모델을, 여러분도 각자의 손으로 각자 나름의 완성에 가깝게 다듬어 보시지 않겠습니까. 저도 작업을 계속해 성과가 나오는 대로 업데이트판을 내놓겠습니다. 여러분의 버전도 Hugging Face나 GitHub에 올려 주시면 기쁘겠습니다.

## 얻게 되는 것

측정된 싱글 스트림 설정——사용자 한 명, 스트림 하나, C=1, thinking
끔——에서 서빙된 모델은 두 대의 DGX Spark 위에서 약 35 tok/s로
응답한다. 정확한 수치는 35.09 tok/s이며, pair 1·NCCL/IB (RDMA)·
MTP 드래프트 K=2 구성을 일본어 산문 프롬프트 64개·
`max_tokens=512`·`temperature=0`의 고정 룰러로 측정했다. 스톡의
기준 행——pair 2·sockets·드래프트 없음·같은 룰러의 처음 8개
프롬프트——는 10.75 tok/s다. 가장 가까운 행은 아래 결과 표에
있다(pair 1·sockets·K=2의 stock + MTP는 expert parallel 오프로
19.05 tok/s, 같은 route h + MTP K=2 구성의 pair 1·sockets는
expert parallel 온으로 24.01 tok/s——`ep` 열 참조).

## 속도의 근원

| 요소 | 내용 | 측정값 |
|---|---|---|
| 기본 속도 | route h 재양자화(BF16 dense 측을 W4A16 NVFP4로) | 18.98 tok/s, 드래프트 없음——pair 1·sockets·64개 프롬프트 전부(드래프트 없는 RDMA 행은 미측정. stock의 드래프트 없는 행은 pair 2·sockets·처음 8개 프롬프트에서 10.75 tok/s) |
| 투기 디코딩 | 체크포인트 자체의 MTP 헤드, K=2 | 수용률 0.62, 사이클당 약 2.24 출력 토큰 |
| 결합 | 재양자화 + MTP K=2 + NCCL over RDMA | **35.09 tok/s** —— pair 1·RDMA·64개 프롬프트 전부·K=2 |

탐욕 디코딩에서는 투기 단계가 베이스 모델이 선택했을 토큰만 수용하므로
출력 분포가 바뀌지 않는다. 따라서 품질 차이는 재양자화에만 귀속되며,
그것을 측정하는 것이 다음 절이다.

## 비용

재양자화는 속도와 맞바꿔 상한이 있는 품질 차이를 받아들인다. 게이트는
속도 측정 전에 선언한 것이며, 릴리스 구성은 모든 기준을 통과해야 했다.

| 기준 | 임계값 | stock | route h | 결과 |
|---|---|---|---|---|
| 64개 프롬프트 룰러에서의 퇴화 출력 | 0 | 0 | 0 | PASS |
| 홀드아웃 passage의 perplexity 비율 | <= 1.10 | 1.0 | 1.051 | PASS |
| eval-200 실효 정확도(150개 항목, 0/50 tool 플로어 제외) | stock보다 0.02 이상 낮지 않음 | 0.447 | 0.453 (+0.0067) | PASS |
| 약 2000 토큰 probe의 TTFT | stock의 1.2x 이내 | 2.380 s | 2.436 s (x1.02) | PASS |

perplexity 비율 1.051은 stock 대비 5.1% 증가로, 선언한 게이트 안쪽이기는
하지만 동등이 아니라 증가다. eval-200은 200개 항목 원점수에서 +0.005,
게이트가 채점하는 150개 실효 항목에서 +0.0067 움직였다. 이 규모의
세트로는 그렇게 작은 차이를 구분할 수 없으며 그 자체로 동등성을
증명하지 않는다.

두 점수와 각 지표가 실사용에서 의미하는 바를 담은 전체 차이를
`results/quality.tsv`에서 렌더링한 것:

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

## 재현 방법

전체 런북(전제 조건·정확한 명령·예상 소요 시간·디스크 필요량)은
[AGENTS.md](AGENTS.md)에 있다. 요약하면, `requant/requant.py --target h`
로 체크포인트를 재양자화하고, RDMA 이미지와 오버레이를 빌드하고,
`serve/start-head.sh`와 `serve/start-worker.sh`로 서빙하고,
`requant/verify.py check`로 게이트를 통과시킨 뒤 `bench/measure.py`로
측정한다.

## 되지 않았던 것과 시기

같은 룰러에서 측정한 것들이다. 탐색 공간을 기록으로 남기기 위해
나열한다.

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

위와 아래의 측정 시계열이다.

| 날짜 | 이정표 |
|---|---|
| 2026-09-13 | pair 2에서의 stock 베이스라인: C=1로 10.75 tok/s |
| 2026-09-15 | stock + 공식 MTP K=2: pair 1에서 19.05 tok/s, pair 2에서 17.84 tok/s. 대부분의 대안 레버가 이 기간에 실패 |
| 2026-09-16 | route h 재양자화, RDMA 이미지, pair 1에서 헤드라인 35.09 tok/s |

## NVIDIA NVFP4 체크포인트에서 시작한 이유

공식 `nvidia/GLM-5.3-Flash-NVFP4` 릴리스는 NVIDIA 자체 툴체인으로
양자화되었고, 모델 카드는 사실상 정확도 손실이 없음을 보여주는
BF16 대 NVFP4 벤치마크 수치를 공개한다(모델 카드 리비전
`09b04e5e74bca08ca8549fc736d4cdd8624bfde3`): GPQA Diamond
0.9217 -> 0.9211, SciCode 0.5621 -> 0.5769, MMMU Pro 0.7688 -> 0.763,
AA-LCR 0.71 -> 0.7106, IFBench 0.613 -> 0.6054, Terminal Bench 2.1은
0.8258 -> 0.8315다. 이것이 신뢰할 수 있는 기반이 되는 이유다.
나의 route h가 하는 일은 공식 릴리스가 BF16으로 남겨둔 레이어에 같은
4-bit 처리를 확장하는 것뿐이며, 트레이드오프가 계속 보이도록 게이트
수치를 속도 수치 옆에 둔다.

## Stage 1 — 릴리스 구성

- **route `h` 재양자화** — attention linear(KDA fused `in_proj`/`out`
  projection, MLA q/kv/o, indexer `wq_b`), shared-expert의
  `gate`/`up`/`down`, 그리고 `lm_head`를 W4A16 NVFP4로 변환.
  router·norm·embedding은 BF16 그대로다.
- **NCCL over RDMA** — 파생 이미지(`docker/Dockerfile.nccl-ib`)가 공식
  이미지 위에 Ubuntu questing의 `rdma-core`를 설치해 NCCL NET/IB
  플러그인이 init할 수 있게 한다. `serve/nccl-ib.sh`가 노드별 HCA와
  RoCE v2 GID를 도출하고 `uverbs*`/`rdma_cm` 디바이스와
  `IPC_LOCK`/무제한 `memlock`을 전달한다.
- **공식 MTP 드래프트 K=2** — 모델 자체의 MTP 레이어에 대해
  `num_speculative_tokens=2`. `mtp-bf16` 오버레이가 드래프트
  서브트리를 비양자화로 구성해 활성화된다(스톡 이미지는 드래프트가
  타겟의 NVFP4 quant config를 상속하게 만들어 BF16 MTP 가중치 로드에서
  크래시한다).

## 속도 결과

고정 룰러: 일본어 산문 프롬프트 64개, `temperature=0`,
`max_tokens=512`, 빈 assistant continuation으로 thinking을 건너뜀.
특기가 없으면 C=1이 64개 프롬프트 전부를 순차 스트리밍하고 프롬프트별
TTFT/TPOT를 잰다. 동봉된 `serve/` 스크립트는 `--max-model-len 16384`,
`--gpu-memory-utilization 0.85`, FP8 KV 캐시를 고정한다. 다만 측정된
행들은 변형으로 실행됐다——2026-09-13 stock 베이스라인은
`--max-model-len 131072`, c1pair와 route h 행은
`--gpu-memory-utilization 0.86`, expert parallel은 `ep` = `off`로 표시한 stock + MTP 두 행(17.84, 19.05 tok/s)에서는 오프였고 나머지에서는 온이었다. 각 행의 정확한 플래그는 `results/results.tsv`의 `source_log`가 가리키는 `results/logs/` 아래 동봉 파일에 있다.

계산 방법:

- **tok/s** — 패스 전체의 completion token 합계 / 총 wall time.
  TTFT를 포함하는 창 전체 스루풋이다.
- **TPOT med** — 프롬프트별 `(wall - TTFT) / (tokens - 1)`의 중앙값.
- **weighted TPOT** — 프롬프트 전체의
  `sum(wall - TTFT) / sum(tokens - 1)`. 토큰 가중 평균이다.
  TPOT med의 역수는 tok/s와 다른 통계이며 일치를 기대하지 않는다
  (같은 실행에서 51.5 ms -> 19.42 대 실측 18.98).
- **accept** — 패스의 spec-decode 카운터 차분:
  accepted / draft tokens. 헤드라인 행의 사이클당 평균 수용 길이
  (출력 수)는 2.244로, 수용 드래프트 토큰 1.244에 보너스 토큰을 더한
  것이다.
- **quality gate / TTFT gate** — 위의 네 가지 조건 게이트를 행별로 두
  열로 나눠 표시한다. degenerate/PPL/eval-200 레그는 체크포인트에
  붙는다(PASS (PPL x1.051) 셀). TTFT 레그는 transport에 의존하며
  route h의 sockets 서빙에서 한 번만 probe했다. sockets 행은
  "verified on sockets", RDMA 행은 "verified on sockets; not re-run
  under RDMA"로 표시하며, TTFT probe를 거치지 않은 transport에서
  네 조건 PASS를 주장하지 않는다. transport별 TTFT는 각 행의
  TTFT med 열을 보면 된다.

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

† 이 행들은 룰러 전체 64개가 아니라 처음 8개 프롬프트로 C=1을
돌린 것이다.

반복 측정(둘 다 2026-09-16에 확정):

- **두 번째 페어** — 같은 헤드라인 구성을 다른 노드 페어에서
  34.47 tok/s. 동일 구성 반복 실행에서 보는 대략 5-7%의 페어 간
  오프셋 안쪽이다.
- **공개용 재측정** — 겹치지 않는 새 64개 프롬프트 세트에서
  prefix 캐시 비활성·워밍업 별도 계상으로 TCP/RDMA 교대 재실행:
  RDMA에서 34.48과 35.12 tok/s, TCP에서 23.75와 24.35 tok/s.
  RDMA 측 재측정은 원래 35.09의 ~2% 이내다.
- pass별 GPU 텔레메트리(양 노드에서 2 s 간격 nvidia-smi; SM clock
  median / power median–max / temp max): 34.48 pass는 2190 MHz,
  24.9–27.1 W, 68 °C. 23.75 pass는 2190 MHz, 20.5–23.6 W, 64 °C.
  35.12 pass는 2190 MHz, 24.5–26.7 W, 65 °C. 24.35 pass는
  2190 MHz, 20.6–24.3 W, 65 °C. 두 번째 페어 재실행은 텔레메트리
  미채취(n/m)다.

미측정 행:

- **K sweep** — 같은 체크포인트에서 route h + RDMA의 K=0/1/3/4 전체
  스윕. 워밍업·순서 효과를 보기 위해 양방향 K 순서로 큐에 넣었다.
  sweep A의 한 패스는 이미 확정됐다——pair 1·RDMA에서 K=1이
  34.28 tok/s, K=3이 31.98——, 전체 스윕 검수가 끝날 때까지 K=2는
  측정한 값 중 최선이지 입증된 최적이 아니다.

헤드라인 실행(pair 1, 2026-09-16): 923.43 s wall time 동안
32407 completion tokens, 실패 요청 없음, finish reason은
61 `length` / 3 `stop`. stock 체크포인트의 C=32 집계는
94.66 tok/s(sockets)와 109.03 tok/s(RDMA)였고, route h의 C=32 행은
아직 없다.

## 이름에 대하여

와비(Wabi, 侘び)는 불완전하거나 소박한 것을 받아들이고 그 안에서 풍요로움을 찾는 일본의 감각입니다. 이 릴리스는 작업 진행 중인 것을 있는 그대로 공개하며, 개선은 측정으로 확인될 때에만 포함합니다.

## Stage 2

더 빠른 구성이 뒤따를 수 있다.

## 파일

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

`eval-200.jsonl`의 프롬프트 두 건은 공개용으로 인용을 다시 작성했다
(ASCII 안전 인용 / 지시문 재표현). 기대 답과 채점은 바뀌지 않았다.

## Contributors

- tenhkspark(메인테이너) — 걱정하며 지켜보다가, 내라고 말한 사람
  (익명 핸들만 표기, 실명 비공개).
- Claude Fable 5.1 (Anthropic) — 방향, 실험 설계, 리뷰와 검수.
- Claude Opus 5 (Anthropic) — 릴리스 전 리뷰.
- Astra (OpenAI, via pi) — 측정과 계획에 대한 적대적 리뷰.
- Devin SWE-2 (Cognition) — 구현, 진단, 큐 스테이징, 리포지토리
  드래프트 작성.
- GLM-5.3 (Z.ai) — 구현 담당.
- GLM-5.3-Flash (Z.ai) — 구현 담당 및 요약.

## License

Apache-2.0. LICENSE 참조. 모델 가중치는 이 리포지토리에 포함되지
않는다. `requant/requant.py`로 NVIDIA 체크포인트를 각자 재양자화할 것.
