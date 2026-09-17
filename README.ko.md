# glm53-flash-nvfp4-2node

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh.md) · [한국어](README.ko.md)

두 대의 NVIDIA DGX Spark 노드로 `GLM-5.3-Flash-NVFP4`를 단일 스트림 디코드 기준 **35.09
tok/s**로 서빙합니다 — 단일 사용자, 단일 스트림, C=1, 사고 모드
해제 조건 — NCCL/IB (RDMA) 기반 노드 페어 1에서 체크포인트
자체 MTP 드래프트를 K=2로 설정하고, 고정된 기준 척도인 64개 일본어
산문 프롬프트에 대해 `max_tokens=512`, `temperature=0`으로 측정했습니다. 실제로 사용해 보면
일본어 산문이 눈에 띄게 끊겨서 출력되는 대신 화면에서 저가 읽는
속도와 거의 비슷하게 출력된다는 뜻입니다.

이 저장소는 해당 수치를 달성한 레시피입니다: 두 노드에 걸친 Ray 텐서
병렬화 (TP=2), 단일 스트림 디코드 속도를 복원하기 위해 저가 사용하는
가중치 전용 재양자화 (루트 h), RDMA를 지원하는 파생 이미지, MTP
드래프트를 로드할 수 있게 해 주는 오버레이를 포함합니다. 이곳의 모든 것은
저 자신의 구현이며, 저의 자체 하드웨어에서 저가 직접 측정한 결과입니다 — *미완성을 즐기며*.

아래의 모든 수치는 해당 조건(페어, 전송 방식, 프롬프트
개수, K, 전문가 병렬화, `max_tokens`)을 명시하고 있으며, 속도
표의 모든 행은 데이터의 출처가 된 로그명을 명기합니다. 엄밀하게
동일 조건 비교를 완성할 순정(stock) 행은 2026-09-17에 나왔습니다 —
14.39 tok/s, pair 1, RDMA, 드래프트 없음, 64개 프롬프트 전체 — 따라서
재양자화와 드래프트 각각이 단 하나의 조건 변경으로 분리됩니다.

이 레시피는 외부 환경에서도 역으로 재실행되었습니다. 다른 노드 페어의
비어 있는 작업 디렉터리에서 시작하여, 공개된 [AGENTS.md](AGENTS.md)와
`scripts/agent-run.sh`만을 활용해 실행했을 때, 서빙 구성을
완료하고 10개 프롬프트 스모크 테스트에서 37.08 tok/s를 측정했습니다 — 기준치 35.09 대비
+5.7%이며, 함께 제공된 검사기가 허용하는 7% 오차 범위 내입니다. 아래의
"공개된 런북 기반 재현" 항목에서 해당 테스트가 무엇을 보여 주며 무엇을 보여 주지 않는지 설명합니다.

모델 가중치는 포함되어 있지 않습니다. 입력 체크포인트는 NVIDIA
`GLM-5.3-Flash-NVFP4` 릴리스이며, 재양자화 스크립트는 아무것도 재배포하지
않고 해당 체크포인트의 BF16 덴스 선형 레이어만 다시 작성합니다.
스크립트가 여러분이 보유한 사본을 가리키도록 설정하십시오.

몇 가지 개선 아이디어가 여전히 저의 목록에 남아 있지만, 모델이 저가
사용하기에 충분한 수준에 도달했으므로 지금 공개합니다. 이는 현재 진행 중인
작업입니다: 받아 가셔서 여러분 각자가 생각하는 완성형으로 발전시켜 주십시오.
저 또한 작업을 계속하여 결과가 나오는 대로 업데이트를 공개할 것이며,
Hugging Face와 GitHub에서 여러분의 버전도 볼 수 있기를 기대합니다.

릴리스 태그: **v1**. 재양자화된 체크포인트는 Hugging Face에
`tenhkspark/GLM-5.3-Flash-NVFP4-Wabi`라는 이름의 단일 저장소로 동일한
태그에 공개되어 있습니다. v2는 아직 없습니다: 다른 구성이 동일한 검증 관문을
통과하고 동일한 척도에서 더 빠른 속도를 기록할 때에만 v2 태그가 나타날
것입니다.

## 왜 두 개의 노드인가

체크포인트는 디스크 상에서 약 204 GB(route-h 재작성 후 약 190 GB)이며
단일 노드의 128 GB 통합 메모리에 들어가지 않으므로, 모델은 TP=2로 두 노드에
나뉘어 있고 모든 디코드 단계는 두 노드 사이의 링크를 건너갑니다. 두 노드는
경로상에 스위치 없이 ConnectX-7급 포트에 단일 200GbE QSFP 구리 케이블로 포트 대
포트 직결되어 있으며, 여기의 모든 수치는 바로 그 직결된 한 쌍에서 측정되었습니다.
동일한 netdev가 두 전송 방식을 모두 전달합니다: `NCCL_IB=1`은 RoCE v2 기반으로
NCCL을 실행하고(표에서 "NCCL/IB (RDMA)"), `NCCL_IB=0`은 TCP("sockets")로 폴백합니다.

링크가 디코드 루프 내에 위치하므로, 여기서 전송 방식은 사소한 세부사항이 아닙니다
— 아래의 세 가지 레버 중 하나입니다.

## 공개된 구성

- **경로 `h` 재양자화** — 어텐션 선형 계층(KDA 융합 `in_proj`/`out` 투영,
  MLA q/kv/o, 인덱서 `wq_b`), 공유 전문가 `gate`/`up`/`down`, 그리고
  `lm_head`가 W4A16 NVFP4로 변환되었습니다. 라우터, 정규화 계층 및
  임베딩은 BF16으로 유지됩니다.
- **RDMA 기반 NCCL** — 파생 이미지(`docker/Dockerfile.nccl-ib`)가 공식 이미지
  위에 Ubuntu questing `rdma-core`를 설치하여 NCCL NET/IB 플러그인이
  초기화될 수 있도록 합니다; `serve/nccl-ib.sh`는 노드별 HCA 및 RoCE v2 GID를
  도출하고 `uverbs*`/`rdma_cm` 장치와 `IPC_LOCK`/무제한 `memlock`을
  전달합니다.
- **K=2 조건의 공식 MTP 드래프트** — 초안 하위 트리를 비양자화 상태로 빌드하는
  `mtp-bf16` 오버레이를 통해 활성화되어 모델 자체의 MTP 계층에 대해
  `num_speculative_tokens=2`를 적용합니다(기본 이미지는 드래프트가 타깃의
  NVFP4 양자화 설정을 상속받도록 만들어 BF16 MTP 가중치를 로드할 때 충돌합니다).

## 각 변경 사항이 가져오는 이점

솔직한 분해 분석을 위해서는 한 쌍의 노드, 하나의 전송 방식, 하나의 측정 기준에
대한 네 개의 행이 필요합니다. 네 개 모두 측정되었습니다:

| step | what it isolates | row | measured |
|---|---|---|---|
| stock, no draft, RDMA | the starting point on the fast transport | pair 1, 64 prompts | 14.39 tok/s |
| route h, no draft, RDMA | the requant alone | pair 1, 64 prompts | 27.15 tok/s |
| route h + MTP K=2, RDMA | the draft on top of the requant | pair 1, 64 prompts | 35.09 tok/s |
| route h + MTP K=2, sockets | the same checkpoint and draft on the slow transport | pair 1, 64 prompts | 24.01 tok/s |

위의 세 행은 이제 모든 조건 — pair 1, NCCL/IB (RDMA), 동일한 64개 일본어
산문 프롬프트, `temperature=0`, `max_tokens=512`, 전문가 병렬 켬, 사고 끔 — 을
공유하므로, 재양자화와 드래프트 각각이 다른 모든 요소로부터 깔끔하게 분리됩니다:

- stock, no draft: 14.39 tok/s, TPOT 중앙값 69.0 ms, TTFT 중앙값
  0.293 s, 실패한 요청 없음.
- route h, no draft: 27.15 tok/s, TPOT 중앙값 35.7 ms, 가중 TPOT
  36.4 ms, TTFT 중앙값 0.269 s, 실패한 요청 없음.
- route h + MTP K=2: 35.09 tok/s, TPOT 중앙값 27.9 ms, TTFT 중앙값
  0.320 s, 수락률 0.6221, 실패한 요청 없음.

27.15 / 14.39 = 1.89이고 35.09 / 27.15 = 1.29이므로, 이 구성에서
재양자화는 약 1.89배, 드래프트는 그 위에서 약 1.29배의 가치가 있습니다.
이 둘은 이 README에서 단 하나의 조건 변경에 걸쳐 취해진 유일한 두 비율입니다.
다른 모든 행의 쌍은 둘 이상의 항목에서 차이가 나므로, 저는 그것들을 나누지
않습니다. 두 비율 모두 양변이 단일 패스이며, 동일 구성의 실행 간 편차는
약 2%입니다.

오늘 제가 말씀드릴 수 있는 것은, 각각 하나의 쌍과 하나의 전송 내에서 다음과 같습니다:

- **드래프트.** Route h, pair 1, 64 프롬프트: RDMA 상에서 드래프트 없이
  27.15 tok/s, K=2 MTP 사용 시 35.09 tok/s (x1.29); sockets 상에서는
  18.98 및 24.01이며, 네 경우 모두 expert parallel이 켜져 있습니다. K=2
  패스에서의 수락률은 0.6174–0.6221로 나타나며, 즉 드래프트 주기당
  약 2.24개의 출력 토큰입니다.
- **전송.** Route h + MTP K=2, pair 1, 64 프롬프트: RDMA 상의 35.09 대비
  sockets 상에서 24.01 tok/s; 동일한 pair에서 공개를 위한 두 번의
  재실행 측정값은 sockets 상에서 23.75 / 24.35, RDMA 상에서
  34.48 / 35.12로 나타나므로, 전송 방식에 따른 격차가 재현됩니다.
- **재양자화.** 순정(stock) 대 route h, pair 1, 64 프롬프트, 양쪽 모두
  RDMA, 드래프트 없음, expert parallel 켬: 순정 14.39 tok/s 대비
  route h 27.15 (x1.89). 예전의 순정 기준값 — pair 2, sockets,
  `--max-model-len 131072`에서 ruler의 첫 8개 프롬프트, 10.75 tok/s —
  는 이 분모가 되지 못합니다. 이것으로 나누면 세 가지 조건 변경이
  한꺼번에 섞이므로, 저는 나누지 않습니다.

## 치러야 하는 대가

재양자화는 품질을 속도와 맞바꿉니다. 이 절충은 아래의 네 가지 기준에
대해서만 제한되며 — 모두 일본어, single-turn, thinking 끔 — 그 외의
어떤 것에도 적용되지 않습니다. 통과 기준(gate)은 어떠한 속도 측정 전에 선언되었으며,
배포된 구성은 네 가지 모두를 통과해야 했습니다:

| criterion | threshold | stock | route h | result |
|---|---|---|---|---|
| 64개 프롬프트 ruler에서의 퇴행적 출력 | 0 | 0 | 0 | PASS |
| 8개 문장 프로브에서의 perplexity 비율 | <= 1.10 | 1.0 | 1.051 | PASS |
| eval-200 라이브 정확도 (150개 항목, 0/50 도구 바닥 제외) | 순정 대비 0.02 이하로 떨어지지 않음 | 0.447 | 0.453 (+0.0067) | PASS |
| 약 2000 토큰 프로브에서의 TTFT | 순정의 1.2배 이내 | 2.380 s | 2.436 s (x1.02) | PASS |

**Perplexity 측정 방식.** Perplexity는 본 테스트를 위해 작성되었으며
`requant/verify.py`에 인라인으로 정의된 여덟 개의 짧은 일본어 문장
(총 249자)을 대상으로 측정됩니다. 이 문장들은 어떠한 공개 말뭉치에서도
가져오지 않았으며 학습 데이터셋 배제 또한 주장하지 않습니다. 각 문장은
서빙된 엔드포인트로 프리필 전용 요청(`echo=true`, `max_tokens=1`,
`prompt_logprobs=1`, temperature 0이 적용된 `/v1/completions`)으로
전송됩니다. 실제 프롬프트 토큰들의 로그 확률은 여덟 개 문장 전체에 걸쳐
누적되며, 단일 풀링된 해당 토큰 스트림에 대한 평균 음의 로그 가능도에
지수 함수를 취해 하나의 숫자로 산출합니다. Stock과 candidate는 동일하게
서빙된 모델을 대상으로 동일한 함수를 통해 측정되며, 통과 기준은 candidate/stock <= 1.10
비율입니다. 릴리스된 route h는 stock의 11.94 대비 12.54를 기록하여 1.051의
비율을 보였습니다. 표본이 작으므로, 이는 동등성이 아니라 심각한 성능 저하가
없는지를 점검합니다.

**eval-200 채점 방식.** eval-200은 본 저장소에 포함된 200개의 일본어 단일 턴
프롬프트 고정 세트입니다(`bench/eval-200.jsonl`):
reason, trap, tool, longread가 각각 50개씩이며, easy/medium/hard가 섞여 있습니다.
모든 항목은 서비스 중인 엔드포인트에서 탐욕적으로 답변되며(temperature 0,
`max_tokens` 384, 빈 어시스턴트 이어쓰기를 통해 사고 과정을 건너뜀)
`requant/verify.py`의 결정론적 규칙에 따라 채점됩니다:
항목의 루브릭이 정답만을 요구할 때는 공백을 무시하는 완전 일치,
그 외의 경우에는 부분 문자열 포함 여부입니다; trap 항목은 거부 표현을 포함해야 하고
숫자가 없어야 하며, tool 항목은 필요한 함수들을 순서대로 명시하고
예상 최종값을 제시해야 합니다. tool 열은 설계상 모든
체크포인트에서 0/50인데, 프롬프트에서 호출 가능한 함수를 전혀 지정하지
않고 어떤 도구 스키마도 전송되지 않기 때문입니다. 따라서 게이트는 활성 상태인
나머지 150개 항목을 채점하며, 후보 모델이 요청 오류 없이 stock 기준 0.02
이내를 유지할 것을 요구합니다 (0.447 stock -> 0.453 route h). 원시 총합 67/200 ->
68/200은 투명성을 위해서만 보고됩니다. 해당 수치에는 영구적인
영점 열이 포함되어 있으며 합격/불합격 판정 기준이 아닙니다. 범주별 변동은
독립적이지 않습니다. trap 채점기는 거절에 보상을 주므로, 더 회피적인
모델일수록 그곳에서 더 높은 점수를 받고 reason에서는 더 낮은 점수를 받습니다. 순 +1 총합은
방향성 저하에 의해서도 발생할 수 있습니다.

**TTFT 프로브**는 바로 그 동일한 여덟 개의 문장을 열두 번 이어 붙여
하나의 프롬프트로 전송한 것입니다. 게이트는 서빙 중인
route-h 소켓에서 이에 대한 후보의 time-to-first-token을
기존 기본 버전(stock)과 비교합니다.

1.051의 perplexity 비율은 순정 대비 5.1% 증가한 수치로, 선언된 게이트
내부이긴 하지만 동등한 수준이 아니라 증가한 것입니다. Eval-200은
200-item 원점수에서 +0.005, 게이트가 채점하는 150 live items에서는 +0.0067 변동했습니다.
150-item 세트는 95% 신뢰 수준에서 정확도를 대략 +-0.08까지 분해하므로
(쌍체 분석, 약 20%의 항목이 뒤집힌다고 가정), 제가 측정한 +0.0067은
차이가 없는 것과 구별할 수 없으며, 0.05의 실제 퇴행 역시 마찬가지일 것입니다.
0.02라는 게이트 임계값은 이 세트가 분해할 수 있는 수준보다 더 정밀합니다. 즉
순정보다 진정으로 0.06-0.08 더 나쁜 체크포인트도 여전히
과반의 확률로 이를 통과할 것입니다. 이 게이트를 '퇴행 없음'으로 해석하는 것은 잘못되었으며, 이는 오직
큰 퇴행만을 배제할 뿐입니다.

탐욕적 샘플링을 적용한 추측 디코딩은 타깃 모델이
생성했을 토큰만을 수락하도록 설계되었으므로, 원칙적으로 초안은
출력 분포를 이동시킬 수 없으며 품질 델타는
재양자화 자체에만 귀속됩니다. 저는 그 동등성을 측정하지 않았으므로, 이를
측정된 결과가 아닌 본래의 설계 논거로만 취급하십시오.

전체 델타(두 점수와 각 메트릭이 실제 사용에서 의미하는 바 모두)는
`results/quality.tsv`에서 렌더링되었습니다:

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

위의 모든 행은 일본어, 단일 턴(single-turn) 및 사고(thinking) 비활성화 기준입니다. 긴
문맥(long context), 병렬 요청 및 에이전트 도구 사용을 다루게 될 단 하나의 행은
대기 중인 13개 항목 스위트이며, 이것이 반영되기 전까지 해당 차원들은 이 체크포인트에서
테스트되지 않았습니다.

**이 게이트가 측정하지 않는 것.** 위의 모든 숫자는 일본어, 단일 턴, 사고 비활성화,
탐욕적 디코딩(greedy), 2k 토큰 미만의 문맥, 그리고 한 번에 하나의 요청 기준입니다.
저는 영어 또는 다른 어떤 언어에 대해서도, 코드 정확성에 대해서도, 다중 턴 대화에 대해서도,
도구 호출(도구 열은 구조적으로 영점입니다)에 대해서도, 지시 준수(instruction following)에 대해서도,
긴 문맥에 대해서도 — 출하하는 서빙 윈도우는 307200 토큰이지만 위의 프로브는 모두 2k 정도에
들어갑니다 — 안전성 동작에 대해서도, 혹은 사고가 활성화된
경우(이 모델 제품군이 일반적으로 사용되는 방식)에 대해서도 이 체크포인트를 측정한 바가 없습니다.
재양자화는 밀집 선형(dense linear) 및 lm_head를 다시 작성하므로, 바로 그곳들이 이러한 네 가지
프로브로부터 회귀가 숨을 수 있는 위치입니다. 그중 어느 것에든 의존하신다면, 이 체크포인트를 도입하기 전에
직접 측정하십시오. requant/verify.py는 플래그 하나로 다른 프롬프트 세트를 받습니다.

### 더 작은 품질 차이를 원하신다면 루트 g

`requant/requant.py --target g`는 더 보수적인 재양자화를 작성합니다. 동일한 64개 프롬프트 룰러(ruler)와
K=2의 MTP를 적용하여 RDMA를 통한 페어 2에서 측정했을 때, 루트 g는 0.999의 퍼플렉시티 비율에서
29.46 tok/s를 기록한 반면, 동일한 페어, 전송, K 및 룰러에서 루트 h는 1.051에서 34.47 tok/s를
기록했습니다. 게이트의 역할은 품질 차이를 최소화하는 것이 아니라 범위를 제한하는 것이고, 1.051은 측정 전에
제가 선언한 범위 이내였기 때문에 저는 h를 배포했습니다. 속도를 더 작은 차이에 사용하기를 원하신다면,
g는 단 하나의 플래그 변경입니다.

## 재현 방법

사전 요구 사항, 정확한 명령어, 예상 소요 시간(wall time), 디스크 요구량을 담은 전체 런북은
[AGENTS.md](AGENTS.md)에 있습니다. 요약하자면: `requant/requant.py --target h`로 체크포인트를
재양자화하고, RDMA 이미지와 오버레이를 빌드한 다음, `serve/start-head.sh` 및 `serve/start-worker.sh`로
서빙하고, `requant/verify.py check`로 게이트를 확인한 뒤 `bench/measure.py`로 측정하십시오. 프롬프트
세트는 [bench/README.md](bench/README.md)에 기술되어 있습니다.

## 공개된 런북으로부터의 재현

위의 모든 숫자는 레시피를 작성한 바로 그 사람에 의해, 레시피가 작성된 기기에서 측정되었습니다. 이는 모든
속도 주장의 가장 취약한 부분입니다. 그래서 저는 외부로부터 레시피를 다시 실행했습니다. 다른 노드 페어의
빈 작업 디렉토리, 공개된 상태 그대로의 리포지토리 스켈레톤, 유일한 지침인 [AGENTS.md](AGENTS.md)와
`scripts/agent-run.sh`, 그리고 해당 페어의 자체 노드 중 하나를 오퍼레이터 노드로 사용했습니다.
실행 과정에서 RDMA 이미지와 오버레이가 빌드되었고, 자체 두 노드를 위한 `setup.env`가 채워졌으며,
헤드와 워커가 구동되었고, 구성 게이트가 실행되어 측정되었습니다. 제 작업 트리에서 가져온 것은 전혀 없습니다.
한 가지 단계는 건너뛰었는데, 바로 재양자화 자체입니다. 해당 페어에 이미 배치되어 있던 배포된 루트 h
체크포인트가 다운로드를 대신했으므로, 이 테스트가 재현하는 것은 가중치 재작성이 아니라 체크포인트부터
측정된 토큰까지의 모든 것입니다.

- 해당 실행 자체의 10-prompt smoke에서 37.08 tok/s, TPOT 중앙값 26.5 ms,
  수락률 0.6177, 실패한 요청은 없었습니다.
- 35.09 tok/s인 릴리스 행 대비 +5.7%로, `scripts/verify-result.py`가
  쌍 대 쌍 오차에 대해 허용하는 7% 대역 내에 있습니다.
- TTFT 중앙값은 릴리스 행의 0.32 s 대비 0.36 s로 12.5% 더 느려,
  해당 대역을 벗어납니다. 검사기는 TTFT를 게이팅하기보다는 보고하며,
  이 행이 바로 TTFT가 여전히 보고되는 이유입니다.

테스트가 보여주지 않는 점: smoke 통과는 10 prompts이고
릴리스 행은 서로 다른 프롬프트 세트에서의 64이므로, +5.7%는
일대일의 동일 조건 비교가 아닙니다 — 10-prompt 통과는 기준자보다
실행 간 산포를 훨씬 더 많이 수반합니다. 테스트가 실제로 보여주는 것은
공개된 절차를 빈 디렉터리에서 단독으로 따랐을 때, 제가 릴리스한 것과
동일한 인근의 서빙 구성에 도달한다는 점입니다.
거기에 도달하기까지 런북과 드라이버에 대해 일곱 건의 수정이 필요했으며,
이는 아래 "What did not work, and when"에 나열되어 있습니다. 각각은 클린
실행 실패를 통해 발견되었으며, 그중 어느 것도 제 자체 트리 내부에서는 보이지 않았습니다.

## 속도 결과

고정된 측정 기준: 64개의 일본어 산문 프롬프트, `temperature=0`,
`max_tokens=512`, 빈 어시스턴트 연속을 통해 thinking 생략.
별도 표기가 없는 한, C=1은 프롬프트별 TTFT/TPOT와 함께
모든 64개 프롬프트를 순차적으로 스트리밍합니다. 제공되는 `serve/` 스크립트는
`--max-model-len 307200`, `--max-num-seqs 2`, `--gpu-memory-utilization 0.88`,
FP8 KV 캐시로 고정합니다. 아래의 측정된 행들은 그 대신 측정용 설정으로
실행되었습니다 — 거의 모두 `--max-model-len 16384`에 `--max-num-seqs 20`,
2026-09-13 순정 기준선은 `--max-model-len 131072`에 슬롯 32개,
pair 2 순정 RDMA 행은 16384에 슬롯 32개였습니다.
c1pair 및 route-h 행은 `--gpu-memory-utilization 0.86`을 실행했으며,
`ep` = `off`로 표시된 두 개의 순정 + MTP 행(17.84 및 19.05 tok/s)에서는 expert parallel을 끈 상태였고
그 외 모든 곳에서는 켠 상태였습니다. 각 행의 정확한 플래그는
`results/results.tsv`의 해당 `source_log` 항목이 가리키는
`results/logs/` 아래의 파일에 있습니다.

**왜 출하하는 윈도우는 307200이고 표는 그렇지 않은가.** 16384은 측정용
값으로 — MTP를 처음 띄울 수 있었던 길이 — 위의 모든 비교를 거치는 동안
그대로 고정되어 있었습니다. 2026-09-17에 다시 측정한 결과, 공개하는 구성은
`--max-model-len 307200` `--max-num-seqs 2` `--gpu-memory-utilization 0.88`에서
동일한 64개 프롬프트 ruler에 대해 35.07 tok/s를 기록했고, 16384 / 20 / 0.86의
35.09와 비교하면 윈도우를 18.75배로 늘려도 약 2%의 실행 간 편차 안에서
차이를 검출할 수 없습니다. 긴 윈도우의 대가는 속도가 아니라 동시 실행입니다 —
307200에서는 슬롯 20개로 엔진이 뜨지 않으므로 제공 스크립트는 2개를
요청하며, 0.89 활용률은 이 페어에서 부팅이 거부된 적이 있습니다. 윈도우가
들어가는 이유의 하나는 재양자화입니다: 같은 0.88에서 순정 체크포인트는
156672 토큰이 상한이고(부팅을 거부할 때 vLLM이 그 상한을 출력합니다),
route h는 307200에서 부팅합니다. 그 길이에 가까운 프롬프트로는 아무것도
측정하지 않았습니다 — 위의 "이 게이트가 측정하지 않는 것"을 보십시오.

제가 계산하는 방식:

- **tok/s** — 패스의 총 완료 토큰 수 / 패스 전체의 소요 시간(wall time)이며,
  TTFT를 포함합니다. 전체 윈도우 기준 처리량입니다.
- **TPOT med** — 프롬프트 전반에 걸친
  `(wall - TTFT) / (tokens - 1)`의 중앙값입니다.
- **weighted TPOT** — 프롬프트 전반에 걸친
  `sum(wall - TTFT) / sum(tokens - 1)`이며, 토큰 가중 평균입니다. 출하된
  JSON 다섯 건(릴리스 재실행 네 건 + route h 드래프트 없음 RDMA 행)에 이
  값이 명시적으로 기록된 필드로 들어 있습니다. 나머지 아홉 개 행에 대해서는 이
  저장소에 포함되지 않은, 정제 이전 factory 로그의 요청별 기록으로부터
  제가 직접 같은 값을 계산해 해당 행의 출하된 `results/logs/` 항목에
  추가하고 그 산출 방식을 적어 두었습니다. 2026-09-13 기준선은 요청별
  기록을 어디에서도 찾을 수 없었고, 그 자체의 수치(92.5 ms)는 프롬프트에
  대한 단순 평균으로 서로 다른 통계치입니다. 그래서 이 행과, 아직 값을
  도출하지 않은 나머지 행들은 한 열에 세 가지 정의를 섞는 대신 모두
  `n/m`으로 표기합니다. TPOT med의 역수 또한 tok/s와는 다른
  통계치이며 서로 일치할 것으로 기대되지 않습니다(51.5 ms는 산술적으로
  19.42 tok/s이지만, 해당 실행에서 측정된 값은 18.98이었습니다).
- **accept** — 패스 전반에 걸친 spec-decode 카운터 델타: 수락된 /
  드래프트 토큰. 대표 행에서의 평균 수락 길이(사이클당 출력)는 2.244였습니다.
  1.244개의 수락된 드래프트 토큰에 보너스 토큰을 더한 값입니다.
- **quality gate / TTFT gate** — 위의 네 가지 기준 게이트이며, 행당 두 개의
  열로 나뉩니다. degenerate / PPL / eval-200 항목은 체크포인트
  ("PASS (PPL x1.051)" 셀)에 귀속됩니다. TTFT 항목은 전송 방식에 의존하며,
  route-h sockets 서빙에서 한 번만 프로브되었습니다. sockets 행은
  "verified on sockets"로, RDMA 행은 "verified on sockets; not re-run under
  RDMA"로 표시됩니다 — 어떤 행도 TTFT 프로브가 확인한 적 없는 전송 방식에
  대해 네 가지 조건의 PASS를 주장하지 않습니다. 각 행의 TTFT med 열에는
  해당 전송 방식의 수치가 담겨 있습니다.

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

† 저는 64개 전체가 아니라 ruler의 처음 8개 프롬프트에 대해 C=1로 이 행들을 실행했습니다.
 

분해의 route-h no-draft 단계는 2026-09-16에 도달했고
(27.15 tok/s), stock no-draft 단계는 2026-09-17에 도달했습니다
(14.39 tok/s). 둘 다 RDMA 기반 pair 1에서 동일한 64개 프롬프트 ruler 상의
결과이므로, requant에 대한 가속 계수가 필요로 하는 분모는 측정되어 있습니다:
27.15 / 14.39 = 1.89.
 
이 표에는 의도적으로 "vs stock" 열이 없습니다: 열을 두면 모든 행에서 비율을
만들고 싶어지는데, 여기 있는 stock 행 대부분은 다른 pair, 다른 전송 방식
또는 64개가 아닌 8개의 프롬프트에서 측정되었으므로 그런 비율은 세 가지 조건
변경 사항을 하나의 숫자로 합치게 됩니다. 단 하나의 조건만 다른 두 쌍은 위
문단과 "각 변경 사항이 가져오는 이점"에 적어 두었습니다.
 
K 역시 확정되지 않았습니다. 64개 프롬프트 ruler를 사용한 RDMA 기반 pair 1에서,
K=1은 34.28 tok/s로 측정되었고 K=2는 세 번의 pass에 걸쳐
34.48 / 35.09 / 35.12로 측정되었습니다; 동일한 구성에서의 실행 간 편차는 약 2%이므로,
이 실험으로는 K=1과 K=2를 명확히 구분할 수 없습니다. K=3은 동일한
pair 및 전송 방식에서 31.98로 측정되었으며, 수락률은 K=2에서의 0.6193에서
K=3에서의 0.483으로 떨어집니다. 제가 K=2를 출시하는 이유는 제가 측정한
값들 중 최선이기 때문이지, K=1이 배제되었기 때문이 아닙니다.
 
반복 측정 결과 (모두 2026-09-16 도달):

- **두 번째 쌍** — 다른 노드 쌍에서 동일한 대표 구성을 실행했을 때
  34.47 tok/s를 기록하였으며, 이는 동일 구성의 반복 실행에서 제가
  확인하는 대략 5-7%의 쌍 간 편차 범위 내에 있습니다.
- **발행 재측정** — 접두사 캐싱을 비활성화하고 웜업을 별도로 계산하여,
  겹치지 않는 새로운 64개 프롬프트 세트에서 대표 구성을
  인터리브 방식의 TCP/RDMA로 재실행한 결과: RDMA를 통해 34.48 및
  35.12 tok/s, TCP를 통해 23.75 및 24.35 tok/s를 기록하였습니다. RDMA
  재실행 결과는 원래 35.09의 약 2% 범위 내에 있습니다.
- 각 패스 실행 시간 동안 두 노드 모두에서 `nvidia-smi`로 2 s마다
  샘플링한 패스별 GPU 텔레메트리(노드당 523–772개 샘플)입니다. 네 번의
  모든 패스 전 샘플에서 SM 클럭은 2190 MHz였습니다. 전력은 네 번의
  패스 전체에서 20–27 W 범위에 머물렀고, GPU 온도는 최고 68 °C를
  넘지 않았습니다. 각 패스가 그 범위 안에서 정확히 어디에
  위치했는지는 패스별로 남겨두지 않았습니다. 이 샘플별 CSV 파일들은
  본 저장소에 포함되어 있지 않으며, `results/logs/` 아래의 네 가지
  재실행 JSON 파일들이 포함되어 있습니다.

대표 실행(pair 1, 2026-09-16)은 실패한 요청 없이 923.43 s의
실행 시간 동안 32407개의 생성 토큰을 생성하였으며, 출하된 로그에는
요청별 종료 사유가 아닌 합계만 기록되어 있습니다. 다만 정제 이전의
factory 로그로부터 직접 세어 보니, 대표 실행 자체의 64개 프롬프트에
걸친 종료 사유 구성도 `length` 61개 / `stop` 3개였습니다. 네 가지
릴리스 재실행은 이 항목을
직접 기록하고 있으며 동일한 구성으로 종료되었습니다 — 즉, 대부분의 프롬프트가 자체적으로
멈추기보다는 512-토큰 상한에 도달하였습니다. 기본 체크포인트에서 C=32 집계는
소켓을 통해 95.08 tok/s(pair 2, 64 prompts), RDMA를 통해
109.03 tok/s(pair 2, 64 prompts)였습니다; 이 두 패스는 모두 슬롯 32개로
실행되었고 출하하는 `--max-num-seqs 2`로는 재현되지 않으므로, 공개
스크립트가 내는 값이 아니라 측정용 값으로 읽어 주십시오. 경로 h에 대한
C=32 행은 아직 없습니다.

### 프롬프트 종류별

64-프롬프트 기준 잣대는 산문입니다. 텍스트의 종류가 수치에 얼마나 영향을 미치는지 보기
위해 저는 공개된 구성에서 별도의 32-프롬프트 세트들을 실행했습니다: 페어
1, NCCL/IB (RDMA), 라우트 h + MTP K=2, `temperature=0`, 전문가 병렬
활성화, 각 세트는 C=1을 자체 패스로 측정했습니다. 이 세트들은
64-프롬프트 기준 잣대가 아니며 그 하위 집합도 아닙니다 — 64 대신 32개의
다른 프롬프트들이므로 — 위의 표에서 제외된 상태로 유지됩니다.

- prose, `max_tokens=512`: 34.87 tok/s, TPOT median 28.1 ms, TTFT
  median 0.319 s.
- prose, `max_tokens=128`: 32.80 tok/s, TPOT median 28.9 ms, TTFT
  median 0.293 s.
- code, `max_tokens=512`: 38.20 tok/s, TPOT median 25.6 ms, TTFT
  median 0.342 s.
- structured output and JSON: measured too; their readings are in the
  paragraph below, without TTFT medians.

구조화된 프롬프트는 512 토큰에서 34.84 tok/s, 128에서 33.51로 실행되었습니다 (TPOT 중앙값 27.4 및 27.6 ms); JSON 형태의 프롬프트는 35.35 및 33.30 (25.0 및 25.2 ms)으로 실행되었습니다; 128 토큰의 코드는 36.08 (25.1 ms)로 실행되었습니다. 네 가지 종류 모두 이 구성에서 32.8과 38.2 tok/s 사이에 위치하며, 모든 패스에서 실패한 프롬프트는 전혀 없었습니다.

동일한 서빙 구성 하에서 산문 측정값인 34.87 tok/s는 64-프롬프트 기준 잣대의
35.09에 안착해 있습니다 — 이는 서로 다른 두 산문 세트 사이에서 제가 원할 법한
일치도입니다. 동일한 구성에서 이 동일한 32-프롬프트 산문 세트에 대해 수행했던
이전 패스는 28.88 tok/s로 측정되었습니다: 해당 패스는 느린 윈도우 내에서
실행되었으며, 수용률이 유지되는 동안 (0.6221 대비 0.6189) 가중 TPOT가
헤드라인 패스의 27.9 ms 대비 34.0 ms였으므로, 그 차이는 초안이 아니라
디코드 주기당 시간 때문입니다. 두 측정값 모두 기록으로 유지됩니다. 그 둘
사이의 편차는 다른 무언가가 이 머신을 건드리고 있을 때 하나의 32-프롬프트
패스가 초래할 수 있는 수준이며, 이는 64-프롬프트 기준 잣대가 재실행 전반에서
보여주는 ~2%보다 더 넓습니다.

## What did not work, and when

동일한 기준 잣대에서 측정되었습니다; 탐색 공간을 기록으로 남기기 위해 나열합니다.

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

위의 클린룸 실행에서 일곱 가지 실패가 더 발생했는데, 이는 모델, 전송 또는
체크포인트가 아닌 런북과 드라이버에서 모두 나타났으며, 빈 디렉터리에서 지침을
그대로 따를 때만 드러나는 종류의 문제였습니다. 그중 다섯 가지는 다음과 같습니다:
런북 자체의 셸이 거부한 `mktemp` 템플릿(끝에 오는 `X`가 세 개 미만임); 드라이버가
작성했으나 결코 소싱하지 않은 서브 구성; 이미지 내 헬퍼가 root 권한으로 생성하여 다음
단계에서 쓸 수 없게 된 디렉터리 내의 파일들; 오퍼레이터 노드와 서빙 헤드가 동일한
머신인 경우로, 스크립트가 자신에게 SSH 연결을 시도하여 처리하려 했던 사례; 그리고
JSON 인자 주변의 따옴표를 삼켜버린 셸 인용 처리입니다. 각각은 실행 과정이 아니라
이곳에 게시된 파일들에서 수정되었습니다.

위와 아래에 제시된 측정값들의 타임라인입니다:

| date | milestone |
|---|---|
| 2026-09-13 | stock baseline on pair 2: 10.75 tok/s C=1 |
| 2026-09-15 | official MTP at K=2 on stock: 19.05 tok/s on pair 1, 17.84 on pair 2; most alternative levers failed in this window |
| 2026-09-16 | route h requant, RDMA image, headline 35.09 tok/s on pair 1 |

## Why I started from the NVIDIA NVFP4 checkpoint

공식 `nvidia/GLM-5.3-Flash-NVFP4` 릴리스는 NVIDIA 자체 툴체인으로 양자화되었으며,
해당 모델 카드는 정확도 손실이 실질적으로 없음을 보여주는 BF16 대 NVFP4 벤치마크
수치를 공개하고 있습니다(모델 카드 리비전 `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`):
GPQA Diamond 0.9217 -> 0.9211, SciCode 0.5621 -> 0.5769, MMMU Pro 0.7688 -> 0.7630,
AA-LCR 0.7100 -> 0.7106, IFBench 0.6130 -> 0.6054, Terminal-Bench 2.1 0.8258 -> 0.8315.
이로 인해 신뢰할 수 있는 기반이 됩니다. 저의 route h는 공식 릴리스가 BF16으로 남겨둔
레이어들에 동일한 4-bit 처리를 확장 적용할 뿐이며, 트레이드오프를 계속 확인할 수
있도록 게이트 수치를 속도 수치 옆에 유지합니다. 해당 카드의 수치들은 표준 공개
벤치마크에서 나온 것이므로, 본 저장소 자체의 엄격한 채점기로 채점된 위의 eval-200
총합과는 비교할 수 없습니다.

## What is left for a v2

이것은 미완성으로 알고 있는 항목들의 목록일 뿐 로드맵이 아닙니다. 저로서는 그 어떤 것의
가치도 예측하지 않으며, 수치를 측정했을 때 이를 공개하겠습니다.

**Speed, still open**

- 저 자신의 NVFP4 MoE 커널. 이는 서비스된 요청에서 실제로 실행된 적이
  전혀 없습니다. 세 번의 시도에서 GB10 빌드가 인스턴스화하지 않는 클래스를
  후킹했고, 그 다음에는 잘못된 클래스를, 그 다음에는 예외를 발생시키고 조용히
  오버레이를 비활성화한 호스트 측 호출을 후킹했습니다. 엔진 로그에 생존 증명
  라인과 함께 실행되기 전까지는 측정할 것이 없습니다.
- All-reduce. 제가 실행한 단일 스텝 기여도 분석(eager, stock, single
  stream, sockets 위)에서 노드 간 피어 대기는 약 87 ms의 디코드 스텝
  중 20.8 ms였습니다. RDMA 환경에서 같은 기여도 분석을 다시 한 적이
  없습니다 — 한 번 시도했지만 부팅 중에 죽어서 스텝 데이터가 전혀
  나오지 않았습니다 — 그래서 이 대기 시간이 빠른 전송 방식에서 얼마나
  남는지는 모릅니다. RDMA 행이 tok/s가 더 높은 것은 패스 전체를 놓고
  본 실측치이며, 이 항목에 대한 기여도 분석은 아닙니다. 집합 통신 연산
  횟수 자체를 줄이려는 시도는 하지 않았습니다.
- 학습된 드래프트 모델. 체크포인트 자체의 MTP 헤드는 자신이 제안하는 토큰의
  0.62를 수락합니다. 이 모델 자체의 출력으로 학습된 드래프트는 더 많은 토큰을
  수락할 수 있으며, 이는 매 사이클을 단축합니다. 저의 첫 번째 시도는 데이터가
  부족한 상태이며(약 3M 토큰 레시피 목표 대비 594개의 학습 레코드, 약 301k 토큰,
  홀드아웃 슬롯당 적중률 0.03-0.08), 측정할 가치가 생기기 전에 대략 열 배의
  데이터가 필요합니다.
- 추측 깊이(Speculation depth). K=1 (34.28) 및 K=2 (35.09)는 본 실험의
  실행 간 편차로 구분되지 않으며, K=3은 더 느립니다. 더 나은
  드래프트는 해당 최적점이 위치하는 지점을 변화시킬 것입니다.

**아직 측정되지 않은 품질**

- 영어 및 기타 언어. 본 문서의 모든 프로브는 일본어입니다.
- 코드 정확성. 코드 프롬프트는 속도 표에 나타나며 품질 표에는 전혀
  나타나지 않습니다.
- 다중 턴 대화, 도구 호출, 지시 수행, 긴 컨텍스트 — 출하하는 윈도우는
  307200 토큰이지만 그에 가까운 프롬프트로는 아무것도 측정하지
  않았습니다 — 안전 동작, 그리고 사고 모드가 활성화된 모델(이 제품군이
  통상적으로 사용되는 방식)입니다.
- 신뢰 구간을 포함한 더 큰 코퍼스에서의 Perplexity. 현재의 프로브는 여덟 개의
  문장으로 이루어져 있고 코드가 토큰별 값을 보존하지 않으므로, 비율에 연계된
  신뢰 구간이 존재하지 않습니다.
- eval-200에 대한 대응 표본 검정(paired test). 채점기가 항목별 결과를 폐기하므로,
  실제로는 쌍을 이루는 동일한 150개 항목을 쌍으로 검정할 수 없습니다.

**아직 완료되지 않은 재현**

- 클린룸 실행에서는 다운로드와 재양자화 자체를 건너뛰었습니다.
  가중치가 이미 노드에 존재했습니다. 빈 디스크에서 시작하여
  약 204 GB 다운로드와 약 30분간의 재양자화를 거치는 전체 경로는,
  누구에 의해서도 종단간으로 실행되지 않았습니다.
- 해당 실행에서는 기준 수치가 나온 64개가 아닌, 10개의 프롬프트를 사용했습니다.

**작동하지 않는다고 제가 이미 알고 있는 것들.** 조밀 선형 레이어(dense linears)에 대한 FP8, 전문가
병렬성 비활성화, 산문에서의 3 이상의 K, 그리고 eager 모드 강제입니다. 그
수치들과 해당 수치들이 측정된 조건들은 위의 실패 표에 있습니다.
이들 중 어떤 것이든 다시 시도하기 전에 확인해야 할 곳은 바로 그곳입니다.

## 끝나지 않는 부분

모델을 빠르게 만드는 데는 하루 동안의 실험이 걸렸습니다. 그 결과를
게시 가능한 형태로 만드는 데는 더 오랜 시간이 걸렸으며, 이는 제가 준비되어 있지 않았던 부분이었습니다.

제가 추가한 모든 검증 단계는 무언가를 찾아냈고, 각각의 발견은
실제 문제였습니다. 표현과 누출에 대한 기계적 검증은 첫 번째 실행에서 통과했기에, 저는
본문의 모든 숫자를 로그로 역추적하는 검증을 추가했습니다. 그중
열네 개는 추적할 수 없었습니다. 저는 이를 수정한 뒤, 사전 맥락이 전혀 없는
검토자에게 전체 내용을 읽어보게 했는데, 주요 비교 수치가 서로 다른 하드웨어에서,
서로 다른 전송 방식을 통해, 서로 다른 수의 프롬프트로 측정한 두 숫자를
나눈 것으로 드러났습니다. 저는 이를 수정한 뒤, 배포된 근거 자료와
표들을 대조해 검증했고 그 과정에서 자체 로그의 엉뚱한 행을 참조한
처리량 수치를 발견했습니다. 저는 이를 수정한 뒤, 양자화 스크립트가 본문에서
주장하는 바와 일치하는지 감사하여 잘못된 체크포인트를 지정할 경우
손상된 초안을 아무런 경고 없이 빌드하는 헬퍼를 찾아냈습니다. 저는 이를 수정한 뒤,
품질 관련 주장에 대해 적대적 관점의 검토를 받게 했고, 네 가지 범주 중
하나의 채점기가 모호하게 표현하는 모델에 보상을 준다는 사실을 알게 되었습니다.
이는 제 수치들이 이동한 방향이기도 했습니다.

그 어떤 것도 부주의에서 비롯된 것은 아니었습니다. 각각은 저마다 다른 방식의
관찰을 필요로 했습니다. 그리고 패턴은 늘 같았습니다. 검증을 추가하고, 무언가를
찾아내고, 그것을 수정하며, 그 수정 작업은 작다는 점입니다. 비용이 드는 것은 검증이 아닙니다.
언제나 한 단계가 더 남아 있다는 사실을 발견하는 것이야말로 비용이 드는 부분입니다.

어느 시점에 이르면 솔직한 선택은 멈추는 것이며, 이는 작업이 완료되었기 때문이
아니라 다음 단계가 배포보다 가치가 떨어지기 때문입니다. 저는
여기서 멈추었습니다. 제가 측정되지 않았음을 알고 있는 사항들은 위에 별도의
섹션으로 나열되어 있으며, 미처 확인할 생각을 하지 못한 사항들이야말로
이것이 최종본이 아닌 v1인 이유입니다. 만약 여러분이 그러한 점을 찾아내신다면, 그것이 바로 시스템이
작동하고 있다는 뜻이며 저는 그 이야기를 듣지 못하기보다는 듣기를 원합니다.

## About the name

와비(侘び)는 불완전하거나 소박한 것을 받아들이고 그 안에서 풍요로움을
찾는 일본의 미의식입니다. 이번 릴리스는 제가 있는 그대로 공개하는
진행 중인 작업이며, 개선 사항들은 측정되는 대로 포함됩니다.

## What comes next

나머지 유형별 세트, 그리고 두 가지 K 순서 모두에서의 완전한
K 스윕입니다. 어떤 구성이 이 기준선에서 35.09 tok/s를 넘어서고
동일한 게이트를 통과한다면, 그것은 v2로 공개됩니다.

## Files

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

두 개의 `eval-200.jsonl` 프롬프트는 공개를 위해 따옴표를 다시 지정했습니다(ASCII 안전
따옴표 지정 / 지시문 문구 수정). 정답과 채점은
그대로 유지됩니다. `bench/prompts-64.jsonl`은 측정한 그대로 제공됩니다 —
[bench/README.md](bench/README.md)를 참조하십시오.

## 역할 분담

설계 결정, 합격 기준, 그리고 이 저장소의 모든 측정치는
저의 것입니다. 구현은 아래 및 [CONTRIBUTORS.md](CONTRIBUTORS.md)에
나열된 AI 좌석들과 함께 수행되었습니다.
게이트, 실행, 수치는 릴리스 전에 제가 확인했습니다.

## 기여자

- tenhkspark (메인테이너) — 걱정하고, 지켜보고,
  진행 신호를 보낸 사람입니다.
- Claude Fable 5.1 (Anthropic) — 방향 설정, 실험 설계,
  검토 및 승인을 담당했습니다.
- Claude Opus 5 (Anthropic) — 릴리스 전 검토를 담당했습니다.
- Astra (OpenAI) — 측정치와 계획에 대한
  적대적 검토를 담당했습니다.
- Devin SWE-2 (Cognition) — 구현, 진단, 실험 큐
  실행, 저장소 초안 작성을 담당했습니다.
- GLM-5.3 (Z.ai) — 구현 좌석입니다.
- GLM-5.3-Flash (Z.ai) — 구현 좌석 및 요약을 담당했습니다.

## 라이선스

Apache-2.0, LICENSE를 참조하십시오. 모델 가중치는 이 저장소의 일부가 아닙니다.
`requant/requant.py`를 사용하여 NVIDIA 체크포인트를 직접 재양자화하십시오.
