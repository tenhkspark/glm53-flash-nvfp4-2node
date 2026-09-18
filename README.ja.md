# glm53-flash-nvfp4-2node

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh.md) · [한국어](README.ko.md)

二台のNVIDIA DGX Sparkノードで `GLM-5.3-Flash-NVFP4` をサービング
し、シングルストリームのデコードで **35.09 tok/s** です——一人の
ユーザー、一本のストリーム、C=1、thinkingオフ。ノードpair 1上で
NCCL/IB (RDMA)を使い、チェックポイント自身のMTPドラフトをK=2で
有効にし、64本の日本語散文プロンプト・`max_tokens=512`・
`temperature=0` の固定ルーラーで測定しました。実際に使うと、日本語
の文章は目に見える塊で届くのではなく、私が画面で読むのとほぼ同じ
歩調で届きます。

このリポジトリは、その数値の裏にあるレシピです。二台のノードに
またがるRayテンソル並列(TP=2)、シングルストリームのデコード速度を
取り戻すために使うウェイトオンリー再量子化(route h)、RDMAが使える
派生イメージ、そしてMTPドラフトをロードさせるオーバーレイ。これらは
私自身の実装であり、私自身のハードウェアでの私自身の実測値です。
ただし同梱ファイルのうち一つは私のものではありません——
`overlays/flashinfer_mla_sparse_sm120.py` はサービングイメージ自身の
vLLMソースファイルで、Apache-2.0ライセンスと上流の著作権ヘッダを
そのまま保ったまま、no-rope MLAに対応させるためにここで改変した
ものです。他のオーバーレイは、同じイメージ自身のソースをビルド時に
書き換える、私が書いたパッチャです。どこまでが上流でどこからが私の
改変かは、[NOTICE](NOTICE)がファイル単位で線を引いています——
*不完全を楽しむ*。

以下の数値にはすべて条件(ペア、transport、プロンプト本数、K、
expert parallel、`max_tokens`)が付いており、速度表の各行は、その値
が出たログの名前を示します。比較を厳密に同条件にするためのstockの行は
2026-09-17に着地しました——14.39 tok/s、pair 1・RDMA・ドラフト無し・
64プロンプト全部。これで再量子化とドラフトのどちらも、条件変化を一つ
だけ挟んで切り分かります。

このレシピは外側からも走らせ直しました。もう一組のノードペアの空の
作業ディレクトリから始め、公開した[AGENTS.md](AGENTS.md)と
`scripts/agent-run.sh` だけを頼りに進めたところ、実行はサービング構成へ
到達し、自前の10プロンプトのスモークで37.08 tok/sを測りました——
35.09のヘッドラインに対して+5.7%、同梱のチェッカーが許す7%の帯の
内側です。そのテストが何を示し、何を示さないかは、下の「公開した
ランブックからの再現」に書いてあります。

モデルの重みは含みません。入力チェックポイントはNVIDIAの
`GLM-5.3-Flash-NVFP4` リリースです。再量子化スクリプトは何かを
再配布する代わりに、そのチェックポイントのBF16 dense linearを
書き換えます。スクリプトは各自のコピーに向けてください。

まだ改善のアイデアはいくつも残っていますが、私が実用と思える水準に
届いたので、今の形で公開します。これは発展途中のものです。どうぞ
手に取って、ご自身の考える完成へ近づけてみてください。私も作業を
続け、結果が出たら更新版を出します。皆さんの版もHugging Faceや
GitHubで見られたら嬉しく思います。

リリースタグ: **v1**。再量子化したチェックポイントはHugging Faceに
`tenhkspark/GLM-5.3-Flash-NVFP4-Wabi` として、同じタグで一つの
リポジトリに公開しています。v2はまだありません。v2タグが付くのは、
別の構成が同じゲートを通り、同じルーラーでより速いと測定できた
場合だけです。

## 二台である理由

チェックポイントはディスク上で約204 GB(route hの書き換え後で
約190 GB)あり、一台のノードが持つ128 GBのユニファイドメモリには
収まりません。そのためモデルはTP=2で二台のノードに分割され、
デコードの各ステップは両者をつなぐリンクを渡ります。二台は
ConnectX-7クラスのポート同士を200GbE QSFPの銅ケーブル一本で
ポート・ツー・ポートで結んでおり、経路上にスイッチはありません。
ここにある数値はすべて、その直結ペアで測ったものです。同じnetdevが
両方のtransportを運びます。`NCCL_IB=1` はNCCLをRoCE v2上で
走らせ(表では "NCCL/IB (RDMA)")、`NCCL_IB=0` はTCPへ
フォールバックします(表では "sockets")。

リンクがデコードループの内側にあるため、ここではtransportは細部
ではありません——下に挙げる三つのレバーの一つです。

## リリース構成

- **route `h` 再量子化** — attention linear(KDA fused `in_proj`/`out`
  射影、MLA q/kv/o、indexer `wq_b`)、shared-expertの
  `gate`/`up`/`down`、そして `lm_head` をW4A16 NVFP4へ変換。
  router・norm・embeddingはBF16のままです。
- **NCCL over RDMA** — 派生イメージ(`docker/Dockerfile.nccl-ib`)が
  公式イメージの上にUbuntu questingの `rdma-core` を入れ、NCCL
  NET/IBプラグインがinitできるようにします。`serve/nccl-ib.sh`
  がノードごとのHCAとRoCE v2 GIDを導出し、`uverbs*`/`rdma_cm`
  デバイスと `IPC_LOCK`/無制限 `memlock` を渡します。
- **公式MTPドラフトK=2** — モデル自身のMTP層に対して
  `num_speculative_tokens=2`。`mtp-bf16` オーバーレイがドラフトの
  サブツリーを非量子化で組み立てることで有効になります(ストック
  イメージはドラフトにターゲットのNVFP4 quant configを継承させ、
  BF16のMTP重みのロードでクラッシュします)。

## 各変更が何をもたらすか

正直な分解には、一つのペア・一つのtransport・一つのルーラー上の
四行が必要です。四行とも測り終えました。

| 段 | 何を切り分けるか | 行 | 実測 |
|---|---|---|---|
| stock, no draft, RDMA | 速いtransport上の出発点 | pair 1, 64 prompts | 14.39 tok/s |
| route h, no draft, RDMA | 再量子化だけの効果 | pair 1, 64 prompts | 27.15 tok/s |
| route h + MTP K=2, RDMA | 再量子化の上に載せたドラフト | pair 1, 64 prompts | 35.09 tok/s |
| route h + MTP K=2, sockets | 同じチェックポイントとドラフトを遅いtransportで | pair 1, 64 prompts | 24.01 tok/s |

上の三行は、いまやすべての条件を共有しています——pair 1、
NCCL/IB (RDMA)、同じ64本の日本語散文プロンプト、`temperature=0`、
`max_tokens=512`、expert parallelオン、thinkingオフ。そのため
再量子化とドラフトのどちらもきれいに切り分かります:

- stock、ドラフト無し: 14.39 tok/s、TPOT中央値69.0 ms、TTFT中央値
  0.293 s、失敗リクエストはゼロ。
- route h、ドラフト無し: 27.15 tok/s、TPOT中央値35.7 ms、
  weighted TPOT 36.4 ms、TTFT中央値0.269 s、失敗リクエストはゼロ。
- route h + MTP K=2: 35.09 tok/s、TPOT中央値27.9 ms、TTFT中央値
  0.320 s、受理率0.6221、失敗リクエストはゼロ。

27.15 / 14.39 = 1.89、35.09 / 27.15 = 1.29なので、この構成では
再量子化がおよそ1.89倍、ドラフトがその上でおよそ1.29倍にあたります。
これは、このREADMEで条件変化を一つだけ挟んで取った二つだけの比です。
ほかの行の組はどれも二つ以上が違うので、私は割りません。どちらの比も
両辺が単発のパスで、同一構成の実行ごとのばらつきは約2%です。

今日言えることは、いずれも一つのペア・一つのtransportの内側での
話です。

- **ドラフト。** route h・pair 1・64プロンプト: RDMAでは
  ドラフト無しで27.15 tok/s、MTP K=2で35.09(x1.29)。socketsでは
  18.98と24.01で、四本ともexpert parallelはオンです。K=2のパスの
  受理率は0.6174–0.6221で、ドラフト周期あたり約2.24出力トークンに
  あたります。
- **transport。** route h + MTP K=2・pair 1・64プロンプト:
  socketsで24.01 tok/sに対し、RDMAで35.09。公開用の再実行二本は
  同じペアでsocketsが23.75 / 24.35、RDMAが34.48 / 35.12だった
  ので、transportの差は再現します。
- **再量子化。** stock対route h・pair 1・64プロンプト、どちらも
  RDMA・ドラフト無し・expert parallelオン: stockが14.39 tok/sに対し
  route hが27.15(x1.89)。古いほうのstock参照——pair 2・sockets・
  ルーラー先頭8プロンプト・`--max-model-len 131072` の10.75 tok/s
  ——はこの分母にはなりません。それで割ると三つの条件変化が一度に
  混ざるので、私は割りません。

## そのコスト

再量子化は品質と引き換えに速度を得ます。その取引に上限が付くのは下の
四つの基準の上だけで——どれも日本語・シングルターン・thinkingオフ
です——その外側については何も保証しません。ゲートは速度測定を始める前に
宣言したもので、リリース構成は四つすべてを通過する必要がありました。

| 基準 | 閾値 | stock | route h | 結果 |
|---|---|---|---|---|
| 64プロンプトのルーラー上の退化出力 | 0 | 0 | 0 | PASS |
| 8文のprobeのperplexity比 | <= 1.10 | 1.0 | 1.051 | PASS |
| eval-200の実効精度(150項目、0/50のtoolフロアを除く) | stockより0.02を超えて下がらない | 0.447 | 0.453 (+0.0067) | PASS |
| 約2000トークンのprobeのTTFT | stockの1.2x以内 | 2.380 s | 2.436 s (x1.02) | PASS |

**perplexityをどう測ったか。** perplexityは、このテストのために
書いた八つの短い日本語文(合計249文字)で測ります。文は
`requant/verify.py` の中に直接書いてあり、公開コーパスから取った
ものではなく、学習データからの除外も主張しません。各文はサービング
中のエンドポイントへprefillのみのリクエストとして送ります
(`/v1/completions` に `echo=true`、`max_tokens=1`、
`prompt_logprobs=1`、temperature 0)。実際のプロンプトトークンの
対数確率を八文すべてにわたって積み上げ、その一本にまとめた
トークン列上の平均負対数尤度を指数化して一つの数値にします。stockと
候補は、同じサービング中のモデルに対して同一の関数で測ります。
ゲートは候補/stockの比が <= 1.10であることで、リリースした
route hはstockの11.938に対して12.543、比1.051でした。サンプルは
小さいので、これは同等性ではなく大きな劣化の有無を見るものです。

**eval-200をどう採点したか。** eval-200は、このリポジトリに同梱
した200本の日本語シングルターンプロンプトの固定セットです
(`bench/eval-200.jsonl`)。reason・trap・tool・longreadが各50本で、
easy/medium/hardを混ぜてあります。各項目はサービング中の
エンドポイントで貪欲に回答させ(temperature 0、`max_tokens` 384、
空のassistant継続でthinkingをスキップ)、`requant/verify.py` の
決定的な規則で採点します。項目のルーブリックが答えだけを求める場合は
空白を無視した完全一致、それ以外は部分文字列の包含です。trap項目は
拒否を含み数字を含まないこと、tool項目は必要な関数を順に挙げ、
さらに期待される最終値を挙げることが条件です。tool列は作りからして
どのチェックポイントでも0/50になります——プロンプトは呼び出し
可能な関数を一度も名指しせず、toolスキーマも送らないからです。
そのためゲートは残る150の実効項目を採点し、候補がstockから
0.02以内に留まり、リクエストエラーがゼロであることを求めます
(stockの0.447 -> route hの0.453)。素点の合計67/200 -> 68/200は
透明性のために示すだけです。これは恒久的なゼロ列を含んでおり、
合否の基準ではありません。カテゴリごとの動きは独立ではありません。
trapの採点器は拒否を褒めるので、より言い逃れるようになったモデルは
そこで高く、reasonで低く出ます。合計が+1動くことは、方向のある
劣化からでも起こります。

**TTFTのprobe** は、その同じ八文を十二回つないで一つのプロンプト
として送ったものです。ゲートは、その上での候補の
time-to-first-tokenをstockと比べます(route hのsockets
サービング上で実施)。

perplexity比1.051はstock比5.1%の増加で、宣言したゲートの内側
ではありますが、同等ではなく増加です。eval-200は200項目の素点で
+0.005、ゲートが採点する150の実効項目で+0.0067動きました。
150項目のセットが精度を分解できるのは95%信頼でおよそ+-0.08まで
(対応あり、項目の約20%が入れ替わると仮定)なので、私が測った
+0.0067はゼロと区別できません——0.05の本物の後退も同じく区別でき
ません。0.02というゲートの閾値は、このセットが分解できる細かさより
細かいのです。この+-0.08の半幅を正規分布の約0.041というσとして
読むと、stockより本当に0.06悪いチェックポイントでも約16%は正答率の
条件を通り、本当に0.08悪いものでも約7%は通ります。この計算が言って
いないことが二つあります: 0.06と0.08は正答率のパーセントポイントの
差であって相対の低下率ではないこと(0.447から0.06下がれば0.387で
あり、6%の低下ではありません)、そして約16% / 約7%は正答率の条件
だけについての正規近似であって、四条件のゲート全体を通る確率では
ないことです。
このゲートはやはり弱いままです——+-0.08の区間は、それが捕まえるはず
の0.02という閾値より四倍広いのです。このゲートを「後退が無い」と
読むのは誤りで、大きな後退を排除するだけです。

貪欲サンプリングでの投機デコードは、ターゲットモデルが出したはずの
トークンだけを受理するように設計されています。したがって原理的には
ドラフトは出力分布を動かせず、品質の差分は再量子化だけに帰属します。
私はその同等性を測っていないので、これは測定結果ではなく設計上の
主張として受け取ってください。

両方のスコアと、各指標が実用上どういう意味を持つかを含む完全な
差分を、`results/quality.tsv` からレンダリングしたものです:

<!-- quality:start -->
| metric | stock | route h | what it means for a user |
|---|---|---|---|
| Perplexity, 8-sentence Japanese probe | 11.938 | 12.543 (ratio 1.051 = +5.1%) | How surprised the model is by the probe text; +5.1% is a real but small regression inside the declared 1.10 gate. The probe is 249 characters of Japanese written for this test -- nothing was held out from training, and a sample this small checks for gross degradation, not for parity. |
| eval-200: reason | 32/50 | 27/50 | Multi-step reasoning items solved; -5 of 50 on a 50-item column -- too few items to separate a real regression from noise. |
| eval-200: trap | 13/50 | 16/50 | Trick-question resistance, graded mechanically: an answer passes if it contains a refusal marker and no digits. That rule rewards hedging, so a checkpoint that became more evasive would gain here while losing on reason -- which is the direction this pair of columns actually moved (+3 trap, -5 reason). I did not test whether the two moves share that cause; do not read the +1 net total as 'no change'. |
| eval-200: tool | 0/50 | 0/50 | Tool-call items score zero on both checkpoints -- the prompts never name a callable tool, so the column is 0 by construction and cannot judge either side. |
| eval-200: longread | 22/50 | 25/50 | Long-document comprehension; +3 of 50 on a 50-item column -- same limit applies. |
| eval-200: total | 67/200 | 68/200 (+0.005 acc) | Overall accuracy moved +0.005 raw (+0.0067 on the 150 live items the gate scores, tool floor excluded), inside the declared gate; by itself it does not prove equivalence. The 95% interval on this difference is about +-0.08, which is four times wider than the 0.02 gate threshold. |
| TTFT, ~2000-token probe | 2.380 s | 2.436 s (x1.02) | Delay before the first token on a long prompt. The 2% gap is the median of three runs of the same prompt and is the same size as the run-to-run spread I measure on identical configurations, so this probe shows no TTFT regression it could have detected -- it does not show that TTFT is unchanged. No user-perception test was run. |
| Degenerate outputs, 64-prompt ruler | 0 | 0 | Empty or looping completions; zero on both sides. Zero out of 64 is consistent with a true rate of up to about 5% (rule of three), and the detector only catches empty output, repeated-token runs and exactly periodic loops -- it cannot see a fluent answer that is wrong, truncated or off-topic. |
| 13-item evaluate suite | -- | -- | The end-to-end serve evaluation (short/long decode, parallel-4, agentic tool-use, long-context, trick questions). It ran on both node pairs on 2026-09-16 and both runs finished, but 4 of the 13 items produced no result on either run: the two agentic tool-use items (3-step tool success, 4th-step final answer) and both 108K long-context items (en, ja) came back blank. The two runs were also not the same configuration -- pair 1 ran with speculative decoding off, pair 2 ran with it on at a 78.0% acceptance rate -- so the 9 items that did produce numbers cannot be read as a route h vs stock comparison, and I do not report them as one here. Agentic tool use and 108K long context therefore remain untested. |
<!-- quality:end -->

上のどの行も日本語・シングルターン・thinkingオフです。長文脈・並列
リクエスト・エージェント的なtool利用を覆う唯一の行——13項目
スイート——は2026-09-16に両ノード対で実行済みですが、13項目のうち
4項目がどちらの実行でも結果を出しませんでした:agentic toolの成功、
agenticの最終回答、統合課題 en 108K、統合課題 ja 108Kです。つまり、
エージェント的なtool利用と108Kの長文脈はこのチェックポイントでは
依然として未測定です——スイートが未実行だからではなく、実行しても
それらを覆う項目が結果を出さなかったからです。両方の実行はroute h対
stockの同条件比較にもなっていません:pair 1は投機なし、pair 2は
投機的デコードの受理率78.0%で走っているので、結果が出た項目でさえ、
二つのチェックポイントの差としては読めません。

**このゲートが測っていないもの。** 上のどの数値も日本語・シングル
ターン・thinkingオフ・貪欲・文脈2kトークン未満・一度に一リクエスト
です。このチェックポイントについて、英語やほかの言語、コードの正しさ、
複数ターンの会話、tool呼び出し(tool列は構造上のゼロです)、指示追従、
長文脈での品質——「長文脈、実測」の針プローブは194,544トークンまで
40問中40問ですが、あれは検索であり、上の品質probeはどれも2kほどに
収まります——安全性の挙動、そして
thinkingを有効にした状態——このモデル族が普段使われる形——の
いずれについても、私は測定を持っていません。再量子化はdense linearと
lm_headを書き換えるので、そこはまさに、この四つのprobeから後退が
隠れうる場所です。どれかに依存するなら、このチェックポイントを採る前に
自分で測ってください。requant/verify.pyはフラグ一つで別のプロンプト
集合を取ります。

### 品質差分をより小さくしたい場合のroute g

`requant/requant.py --target g` は、より保守的な再量子化を書き出し
ます。pair 2・RDMA・MTP K=2・同じ64プロンプトのルーラーで、
route gはperplexity比0.999で29.46 tok/s、同じペア・transport・
K・ルーラーでのroute hは1.051で34.47 tok/sでした。ゲートの
役目は品質差分を最小にすることではなく上限で抑えることであり、
1.051は測定前に宣言した上限の内側なので、私はhを出しました。
速度を渡して差分の小ささを取りたいなら、gはフラグ一つの変更です。

## 立てる前に: 認証は無く、全インターフェースで待ち受けます

`serve/start-head.sh` はAPIを `--host 0.0.0.0` で上げるので、ノードの
全インターフェースで待ち受けます。サーバはAPIキーを検査しません——
ポートに届くクライアントは誰でもリクエストを送れます。このファイルの
残りは、あなたがこれを読んだ前提で書いてあります。

- この構成は信頼できるネットワークの内側に留め、ポートを公開
  ネットワークへ晒さないでください。
- アクセス制御は各自で用意するものです。ノードの外から届く必要が
  あるなら、自分のファイアウォール規則を前段に置き、通信に必要なら
  認証とTLSを持つプロキシも置いてください。
- Rayの管理ポートと、二台のノードが互いに使うポートはAPIのポート
  ではなく、それぞれ別に隔離が要ります。出荷スクリプトはRayの
  ダッシュボードを127.0.0.1に束縛していますが、それは複数あるうちの
  一つのポートにすぎません: APIの前に認証を置いてもRayは保護され
  ません。
- この構成は十分に長いプロンプトでホストのメモリを枯渇させられます
  (どこでそうなるかは下の「長文脈、実測」で測っています)ので、開いた
  ままのエンドポイントは、読まれる経路であるだけでなくノードを
  落とす経路です。

## 再現方法

完全なランブック(前提条件・正確なコマンド・予想所要時間・ディスク
必要量)は[AGENTS.md](AGENTS.md)にあります。要点だけ言うと、
`requant/requant.py --target h` でチェックポイントを再量子化し、
RDMAイメージとオーバーレイをビルドし、`serve/start-head.sh` と
`serve/start-worker.sh` でサービングし、`requant/verify.py check`
でゲートを通し、`bench/measure.py` で測定します。プロンプト集合に
ついては[bench/README.md](bench/README.md)に書いてあります。

**ヘッドラインの速度を再現するには `MTP_DIR` の設定が要ります。**
35.09 tok/sの行はroute hにMTPドラフトをK=2で載せたもので、ドラフトは
`MTP_DIR` がドラフトのディレクトリを指しているときにだけロードされ
ます。`serve/serve.env.example` はその行をコメントアウトのまま、値も
`CHANGEME` で同梱しています——パスは機械ごとに違うので意図的に
そうしてあります——そのため例のファイルを出荷されたまま使うと
サーバはドラフト無しで立ち、ドラフト無しの速度でデコードします:
同じパスでドラフトありが34.99 tok/s(TPOT中央値27.9 ms、TTFT中央値
0.309 s、受理率0.6223、64件中失敗0件)だったのに対し、そのやり方では
28.28 tok/sでした。上の27.15のドラフト無しの行と+4%で整合します。
測定の前に、その行のコメントを外して自分のパスを入れてください。
ドラフトのディレクトリはstockチェックポイントから
`requant/build-mtp-draft.py` が作ります。その手順は
[AGENTS.md](AGENTS.md)にあります。

## 公開したランブックからの再現

上のどの数値も、レシピを書いた本人が、レシピを書いたその機械で測った
ものです——速度の主張として最も弱いところです。そこでレシピを外側から
走らせ直しました。もう一組のノードペアの空の作業ディレクトリ、公開した
ままのリポジトリの骨格、指示は[AGENTS.md](AGENTS.md)と
`scripts/agent-run.sh` だけ、そしてそのペア自身のノードの一つを操作
ノードにしました。実行はRDMAイメージとオーバーレイをビルドし、自分の
二台ぶんの `setup.env` を埋め、headとworkerを立ち上げ、構成ゲートを
通して測定しました——私の作業ツリーからは何も持ち込んでいません。一段
だけ飛ばしています。再量子化そのものです。そのペアにすでに置いてあった
リリース版のroute hチェックポイントがダウンロードの代わりになったので、
このテストが再現するのはチェックポイントから測定されたトークンまでで
あって、重みの書き換えではありません。

- 実行自身の10プロンプトのスモークで37.08 tok/s、TPOT中央値
  26.5 ms、受理率0.6177、失敗リクエストはゼロ。
- 35.09 tok/sのリリース行に対して+5.7%で、
  `scripts/verify-result.py` がペア間オフセットに許す7%の帯の内側です。
- TTFT中央値は0.36 s、リリース行の0.32 sに対して12.5%遅く、この帯の
  外側です。チェッカーはTTFTをゲートにせず報告に留めており、この行が
  まさに、いまも報告している理由です。

このテストが示さないこと: スモークのパスは10プロンプト、リリース行は
64で、しかも別のプロンプト集合なので、+5.7%は同条件の比較では
ありません——10プロンプトのパスは、ルーラーよりずっと大きな実行ごとの
ばらつきを抱えます。示すのは、公開した手順をそれだけに従って空の
ディレクトリから辿れば、私が出したものと同じ近傍のサービング構成へ
届く、ということです。そこへ着くまでにランブックとドライバへ七つの修正が
要りました。下の「うまくいかなかったものと時期」に挙げてあります。
どれもクリーン実行の失敗が見つけたもので、私自身のツリーの内側からは
一つも見えていませんでした。

## 立てたサーバの使い方

`serve/start-head.sh` は `PORT`(既定 8000)にOpenAI互換のAPIを
載せます。このAPIの三つの性質はserve行が決めていて、外からは
推測できないので、ここに書いておきます。

**モデルIDは `GLM-5.3-Flash-NVFP4-Wabi` です。** スクリプトが
`--served-model-name` を渡すので、リクエストに載せるのは——
チェックポイントのパスではなく——この文字列です。ほかのIDは404で
返ります。この確認は机上の話ではありません: 2026-09-17、旧名
`GLM-5.3-Flash-NVFP4` を指したままのクライアントが、
`GLM-5.3-Flash-NVFP4-Wabi` へ指し直すまで全リクエストで404を
受け取りました——served名を一致させることがすべてで、別途たどる
IDとチェックポイントの対応表はありません。`/v1/models` はこのIDを
サービング窓とともに返します:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

headノードからの、完結したリクエスト:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "GLM-5.3-Flash-NVFP4-Wabi",
       "messages": [{"role": "user", "content": "Hello."}],
       "max_tokens": 128}'
```

OpenAI互換クライアントに要る設定は三つだけです: ベースURL
`http://127.0.0.1:8000/v1`(クライアントが別のマシンで動くなら
localhostのところをheadのアドレスに)、モデル
`GLM-5.3-Flash-NVFP4-Wabi`、そしてAPIキーは任意の文字列——サーバは
検査しません。私はまさにその設定にしたコーディングエージェントから、
このサーバを使っています: pi を、ベースURL
`http://<head-node-ip>:<PORT>/v1`、モデル
`GLM-5.3-Flash-NVFP4-Wabi` で向けると、構造化された `tool_calls` が
返り、thinkingは回答へ漏れずに専用フィールドへ収まります。

**ツール呼び出しは有効です。** serve行は
`--enable-auto-tool-choice --tool-call-parser glm45` を持っているので、
クライアントは `tools` を送れて、構造化された `tool_calls` が返ります。
この二つのフラグが無いと、`tools` を載せたリクエストはHTTP 400で
断られます:

```
"auto" tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set
```

**`content` だけでなく `reasoning` も読むこと、そしてthinkingを
切ろうとしないこと。** `--reasoning-parser glm45` はモデルのthinkingを
メッセージ上の別フィールド `reasoning` へ移し、`content` には回答を
残します。オフスイッチに見えるつまみは、オフスイッチではありません:

| クライアントが送るもの | `reasoning` | `content` |
|---|---|---|
| 何も送らない——既定 | thinking | 回答 |
| `chat_template_kwargs: {"enable_thinking": false}` | 空 | thinkingと回答が地続き |
| `continue_final_message` 付きの空のassistantターン | 応答全体 | 空 |

`enable_thinking: false` が止めるのはモデルではなくパーサです。モデルは
thinkingを続け、その思考は読み手の目に触れる `content` に落ちます。
空assistantの継続——ルーラーでthinkingをスキップするために
`bench/measure.py` が使う流儀——は同じ仕組みの裏側です: テンプレートが
thinkingブロックを自分で閉じるので、モデルは閉じタグを一度も出さず、
パーサは応答全体を `reasoning` に残します。どちらもサーバ側の設定では
直りません。どちらも送らず、両方のフィールドを読めば、素のOpenAI互換
クライアントはそのまま動きます。

**窓を使い切らせないこと。** 後述の「長文脈の実測」には、
`--gpu-memory-utilization` や `--max-num-batched-tokens` を変えても
関係なくエンジンごと落ちた4件のプロンプトを記録しています。窓が
ほぼ埋まった状態では、この落ち方はプロンプトの工夫では避けられません。
エージェントの会話は、サーバに強制終了させられる前に、
<!-- SAFE-WINDOW-TBD --> トークンに達する十分手前でコンパクト化する
かクリアしてください。安全な閾値そのものはまだ実測していません。

**実際に何本同時に走るかはフラグの値そのものではありません。** serve行の
`--max-num-seqs 20` はスケジューラが受け入れてよい上限であって、20本が
同時に走る保証ではありません。KVキャッシュプールがその本数を保持できない
ときは、vLLMはリクエストを失敗させる代わりに、入る本数までプリエンプト
します。このプロンプト長でこの構成が実際に何本の同時リクエストを
支えられるかはまだ実測していません: <!-- CONCURRENCY-TBD -->。

**起動のたびに、ホストの空きメモリを確認してください。**
`serve/check-headroom.sh` を両ノードで、最初のリクエストを送る前に。
速度表からは決して分からない運用手順なので、独立した段落を割きます。

GB10は統合メモリです。GPUはホストのRAMから確保し、その予約のうち約100GBは
標準のカーネルカウンタのどれにも現れません——この部品では `nvidia-smi` の
FB Memory Usage がN/Aを返します——ので、`MemAvailable` が唯一の正直な
物差しです。vLLMはプロファイル時にたまたま空いていた量から予算を決めるため、
**余裕を決めるのはフラグではなく、その回の起動**です。無改造の同じ
スクリプトが同じノードで、3回の起動でそれぞれ5470・9728・10270 MiBの
空きでREADYに達しました。見える範囲では何も違わない起動の間に、4.8 GiBの幅が
あります。

そしてREADYの**後**に、さらに約4.7 GiBが使われます。21k→32k→64k→128k→
197,485トークンとプロンプトを伸ばしながら両ノードの `MemAvailable` を
1秒ごとに採ると、最良の起動は10270 MiB（8.24%）から5579 MiB（4.48%）まで
落ちて、そこで止まりました。これはリークではなく一度きりの高水位です——
同じ長さを繰り返しても追加の消費はありません——が、**より大きいprefillの形が
初めて来たときに課金される**。コーディングエージェントが文脈を伸ばしながら
やることが、まさにこれです。その起動は宣言している窓いっぱいを含めて
梯子を全段こなし、失敗はゼロでした。

5470 MiBの起動はそうなりませんでした。6時間動いたあと、20,915トークンの
エージェントのプロンプトの途中で殺されました。ドライバが先に
`NV_ERR_NO_MEMORY` を記録しています。自分の立ち上がりぶんの余地が、最初から
無かったのです。

つまり床は「立ち上がりに要る分＋自分を刈り取るものの線」であり、確認は
起動時にやる。床を割ったノードは調整するのではなく再起動してください。
`--gpu-memory-utilization` を下げるのは見た目ほど効く手ではありません——
204800では律速rankのKVが約3.5 GiBしかなく、utilizationの0.01が1.2 GiBなので、
2段下げれば宣言している窓をもう開けられません。再現する人向けの警告を
ひとつ: **DGX OSは `earlyoom` も `systemd-oomd` も積んでいません。**
私のノードでは自分で入れたOOMデーモンが、これを行儀のよい
プロセス終了に変えていました。無ければ、このハードでは同じ圧力が
ノードのハングになります。

## 速度結果

固定ルーラー: 64本の日本語散文プロンプト、`temperature=0`、
`max_tokens=512`、空のassistant継続でthinkingをスキップ。
特記が無ければC=1が64プロンプトすべてを逐次ストリームし、
プロンプトごとのTTFT/TPOTを取ります。同梱の `serve/` スクリプトは
`--max-model-len 204800`、`--max-num-seqs 20`、
`--gpu-memory-utilization 0.85`、`--enable-prefix-caching`、
FP8 KVキャッシュ、そして両ノードでの
`RAY_memory_usage_threshold=0.99` を固定します。
下の測定行はそうではなく測定用の設定で走っています――ほとんどが
`--max-model-len 16384`・`--max-num-seqs 20`、2026-09-13のstock
ベースラインは `--max-model-len 131072` でスロット32本、pair 2の
stock RDMA行は16384でスロット32本。c1pairとroute hの行は
`--gpu-memory-utilization 0.86`、expert parallelは `ep` = `off` と
記したstock + MTPの二行(17.84と19.05 tok/s)、および末尾のship-script行
(37.33 tok/s)でオフ、末尾のship-script自身のEP-on制御行(35.05 tok/s)を
含む他のすべての行ではオンでした。各行の正確なフラグは、`results/results.tsv` のその行の
`source_log` が指す `results/logs/` 以下のファイルにあります。

**なぜ出荷する窓は204800で、表はそうでないのか。** 16384は測定用の
値で――MTPを最初に立ち上げられた長さです――上のどの比較でも留め置かれ
たままでした。2026-09-17に出荷する窓で測り直すと――この表の他の行と
同じくexpert parallelを有効にした状態で――ルーラーは34.78 tok/s、
TPOT中央値28.1 ms、TTFT中央値0.313 s、リクエスト64件のうち失敗0件
でした。16384 / 20 / 0.86での35.09に対して0.9%の差で、窓は12.5倍に
なっていますが、同一構成での実行ごとのばらつき約2%の中です。この
34.78という数値は `results/results.tsv` に自分の行を持ちません――
内部ランチャでの測定で、そのログは残っていません。下の二つの
ship-script行は `source_log` が `results/logs/` を指しており、この
リポジトリで再現できるのはそちらです。

**出荷するスクリプトはこの行より速い。expert parallelを有効に
しないからです。** `serve/start-head.sh` は
`--enable-expert-parallel` をどこにも渡していませんが、この表の
どの行もそれを有効にして測ったものです。公開しているスクリプトを
そのまま使って同じ64プロンプトのルーラーを回すと、立ち上げ直後の
ペアで **37.33 tok/s** ――TPOT中央値26.4 ms、TTFT中央値0.302 s、
64件中失敗0件、受理率0.6168――、再測で36.95でした。同じスクリプトに
EPの2フラグを戻すと35.05（TPOT 27.9 ms、受理率0.6223）まで落ち、
34.78の行を0.8%以内で再現します。つまり差はexpert parallelだけで、
他には何もありません。EPはここではホストメモリも食います:
READY後に払う高水位が4691 MiBから6014 MiBへ上がり、梯子の低水位が
4.48%から3.38%へ下がりました。このペアでは遅くて大食いなので
スクリプトは出荷せず――上の表も、速い方の数字を借りずに自分が
測った条件のままにしてあります。

長い窓でも `--max-num-seqs 20` でサーバは立ち上がります:
204800ではREADYに達します(909秒、そのあと同じルーラーで失敗0件)が、
307200では立ち上がりませんでした。これは起動時に受け入れられたと
いう事実であって、受け入れられた20本のスロットのうち実際に何本を
同時に動かせるかの実測ではありません——それを決めるのはKVプールで、
その本数は<!-- CONCURRENCY-TBD -->です。下げた
フラグは利用率だけです――0.89はこのペアで起動を拒否された実績が
あり、0.88ではheadノードのホストRAMの空きが約2.1%で、何が先に刈るかに
関わらず枯渇に近すぎます。そこでスクリプトは0.85を出荷し、8.6%を
残します。窓が入る理由の一つは再量子化です: 0.88で公式
チェックポイントは156672トークンが上限(起動を拒否するときvLLMが
その上限を印字します)、route hは204800で起動します。窓を実際に使う
プロンプトで何が出るかは、下の「長文脈、実測」で測っています。

数え方:

- **tok/s** — パス全体のcompletion token合計 / 壁時計時間合計。
  TTFTを含む、窓全体のスループットです。
- **TPOT med** — プロンプトごとの `(wall - TTFT) / (tokens - 1)` の
  中央値。
- **weighted TPOT** — プロンプト全体での
  `sum(wall - TTFT) / sum(tokens - 1)`。トークン加重の平均です。
  これを明示的な記録項目として持つのは、出荷済みJSON五本(リリース
  再実行四本+route hドラフト無しRDMAの行)です。残る九行については、
  このリポジトリに含まれない、サニタイズ前のfactoryログにある
  各実行のプロンプトごとの記録から私自身が同じ量を計算し、その行の
  出荷済み `results/logs/` エントリに書き加え、そこに導出方法を
  記しました。2026-09-13のベースラインはプロンプトごとの記録が
  どこにも見当たらず、
  その値(92.5 ms)はプロンプトごとの単純平均で、別の統計です。その
  ため、この行と、まだ導出していない残りの行は、一つの列に三つの
  定義を混ぜる代わりに `n/m` と表示します。TPOT medの逆数もtok/s
  とは別の統計で、一致は期待しません(51.5 msは算術上19.42 tok/s、
  その実行での実測は18.98)。
- **accept** — パス全体のspec-decodeカウンタ差分:
  accepted / draft tokens。ヘッドライン行の周期あたり平均受理長
  (出力数)は2.244で、1.244の受理ドラフトトークンにボーナス
  トークンを足したものです。
- **quality gate / TTFT gate** — 上記の四条件ゲートを行ごとに
  二列へ分けた表示です。degenerate・PPL・eval-200の各脚は
  チェックポイントに紐付きます("PASS (PPL x1.051)" のセル)。TTFT脚は
  transport依存で、route hのsocketsサービング上で一度だけ
  probeしました。sockets行は "verified on sockets"、RDMA行は
  "verified on sockets; not re-run under RDMA" と表示し、TTFT probeを
  通していないtransportでは四条件PASSを主張しません。
  transportごとの数値は各行のTTFT med列にあります。

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
| route h + MTP | 2 | NCCL/IB (RDMA) | 2 | off | 37.33 | 26.4 | n/m | 0.302 | 0.6168 | — |  |
| route h + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 35.05 | 27.9 | n/m | 0.304 | 0.6223 | — |  |

<!-- results:end -->

† これらの行はルーラー全64本ではなく先頭8プロンプトでC=1を
回したものです。

分解のroute h・ドラフト無しの脚は2026-09-16に(27.15 tok/s)、
stock・ドラフト無しの脚は2026-09-17に(14.39 tok/s)着地しました。
どちらもpair 1・RDMA・同じ64プロンプトのルーラーで取ったものなので、
再量子化の速度倍率が分母として必要とする行は測れています:
27.15 / 14.39 = 1.89。

この表に "vs stock" 列が無いのは意図的です。列を置くと全行で比を
作りたくなりますが、ここにあるstock行の多くは別のペア・別のtransport、
あるいは64本ではなく8本のプロンプトで取ったもので、そうした比は
三つの条件変化を一つの数値へ畳み込んでしまうからです。条件が一つ
だけ違う二組は、上の段落と「各変更が何をもたらすか」に書きました。

Kも決着していません。pair 1・RDMA・64プロンプトのルーラーでは、
K=1が34.28 tok/s、K=2は三パスで34.48 / 35.09 / 35.12でした。
同一構成の実行ごとのばらつきはおよそ2%なので、この実験はK=1と
K=2を分離できていません。K=3は同じペア・同じtransportで31.98、
受理率はK=2の0.6193からK=3の0.483へ落ちます。K=2を出すのは、
測定した値の中で最良だったからであり、K=1が排除されたからでは
ありません。

繰り返し測定(いずれも2026-09-16に着地):

- **二つ目のペア** — 同じヘッドライン構成をもう一組のノードペアで
  測ると34.47 tok/s。同一構成の繰り返し実行で見ているおおよそ
  5-7%のペア間オフセットの内側です。
- **公開用の再測定** — ヘッドライン構成を、重複の無い新規64
  プロンプト集合で、prefixキャッシュ無効・暖機は別計上として
  TCP/RDMA交互に再実行しました: RDMAで34.48と35.12 tok/s、
  TCPで23.75と24.35 tok/s。RDMA側の再実行は元の35.09の
  ~2%以内に収まりました。
- パスごとのGPUテレメトリは、各パスの実行中ずっと両ノードで
  `nvidia-smi` を2 s間隔でサンプリングしました(ノードあたり
  523–772サンプル)。SMクロックは四パスすべての全サンプルで
  2190 MHzでした。電力は四パス全体で20–27 Wの範囲に収まり、GPU
  温度は最高でも68 °Cでした。どのパスが範囲内のどこに位置したか
  というパスごとの内訳は残していません。サンプル単位のCSVはこの
  リポジトリに含めていません。含めているのは `results/logs/` 以下
  の再実行四本のJSONです。

ヘッドラインの実行(pair 1、2026-09-16)は923.43 sの壁時計時間で
32407 completion tokensを出し、失敗リクエストはゼロでした。出荷済み
のログは合計値だけを記録しており、リクエストごとのfinish reasonは
残していませんが、サニタイズ前のfactoryログから数え直したところ、
ヘッドライン自身の64プロンプトの内訳も61 `length` / 3 `stop` でした。
リリース用の再実行四本は同じ項目を直接記録しており、同じ内訳になって
います——つまり
大半のプロンプトは自分で止まらず512トークンの上限に当たって
います。stockチェックポイントでのC=32集計は、socketsで
95.08 tok/s(pair 2、64プロンプト)、RDMAで109.03 tok/s(pair 2、
64プロンプト)でした。この二本はどちらもスロット32本で走っており、
出荷する `--max-num-seqs 20` では再現できません。公開スクリプトが
出す値ではなく測定用の値として読んでください。route hのC=32行は
まだありません。

### プロンプトの種類ごと

64プロンプトのルーラーは散文です。テキストの種類が数値をどれだけ動かす
かを見るため、リリース構成の上で別々の32プロンプト集合を回しました。
pair 1、NCCL/IB (RDMA)、route h + MTP K=2、`temperature=0`、expert
parallelオン、各集合はそれぞれ独立したパスとしてC=1で測定しています。
これらは64プロンプトのルーラーではなく、その部分集合でもありません
——別のプロンプトで、64ではなく32本です——ので、上の表には入れて
いません。

- 散文、`max_tokens=512`: 34.87 tok/s、TPOT中央値28.1 ms、TTFT中央値
  0.319 s。
- 散文、`max_tokens=128`: 32.80 tok/s、TPOT中央値28.9 ms、TTFT中央値
  0.293 s。
- コード、`max_tokens=512`: 38.20 tok/s、TPOT中央値25.6 ms、TTFT中央値
  0.342 s。
- 構造化出力とJSON: これらも測定済みで、値は下の段落にあります
  (TTFT中央値は除く)。

構造化のプロンプトは512トークンで34.84 tok/s、128で33.51(TPOT中央値は27.4と27.6 ms)、JSON形のプロンプトは35.35と33.30(25.0と25.2 ms)、コードの128トークンは36.08(25.1 ms)でした。この構成では四つの種類すべてが32.8から38.2 tok/sの間に収まり、どのパスも失敗プロンプトはゼロです。

散文の値34.87 tok/sは、同じサービング構成での64プロンプトのルーラーの
35.09に重なります——二つの異なる散文集合の間で私が望む一致です。同じ
32プロンプトの散文集合を同じ構成で回した以前のパスは28.88 tok/sでした。
それは遅い窓の中で走っており、weighted TPOTは34.0 msで、ヘッドラインの
パスの27.9 msを上回る一方、受理率は動きませんでした(0.6189に対して
0.6221)。つまり違いはデコード周期あたりの時間であって、ドラフトでは
ありません。どちらの値も記録に残します。二つの開きは、ほかの何かが機械に
触れているときに一回の32プロンプトのパスが出しうる幅で、64プロンプトの
ルーラーが再実行の間で見せる~2%よりも広いものです。

## 長文脈、実測

プローブは `bench/longctx.py` です: 日本語の詰め物を目標の長さまで
伸ばし、先頭・中間・末尾のトークン深度に十個の事実を埋め、それぞれ
一問ずつ聞きます(`temperature=0`)。採点は審判モデルではなく、
コードの完全一致です。セットは `bench/longctx-probe.jsonl` で、
どの文書もその十問の共通prefixなので、長いprefillは各段で一度しか
払いません。出荷構成 — `--max-model-len 204800`、
`--max-num-seqs 20`、`--gpu-memory-utilization 0.85`、
`--enable-prefix-caching`、route hにMTPをK=2で載せてRDMA — での
一度の通しがこれです:

| プロンプトトークン | 見つけた針 | 初回の最初のトークン | キャッシュ後 | prefill |
|---:|---|---:|---:|---:|
| 16,345 | 10/10 | 10.02 s | 4.27 s | 1630.6 tok/s |
| 65,545 | 10/10 | 39.95 s | 2.91 s | 1640.8 tok/s |
| 130,990 | 10/10 | 79.86 s | 3.15 s | 1640.2 tok/s |
| 194,544 | 10/10 | 120.04 s | 4.74 s | 1620.6 tok/s |

40問中40問、深さによる弱いところもありません: 先頭12/12、中間16/16、
末尾12/12。別構成での測定がひとつあり、意図して表から外しています:
`--max-model-len 307200`・利用率0.88 — 公開スクリプトが出すもの
ではありません — で、204,767トークンの文書一本が10/10でした。公開
スクリプトが開かない窓を必要とするので、上の行にはしません。

これは検索であって品質ではありません。194,544トークンでも埋めた事実を
見つけられる、とは言えます。そこで散文や推論が保つかどうかは何も
言えません。

長い窓には制約が付いてきます。

- **窓と同じ大きさのプロンプトは入りません。** プローブの最上段が
  200kではなく190kを狙っているのは実測に基づく理由からです:
  204,754トークンの文書に `max_tokens=64` を足すと204,818で、上限の
  204,800を超え、サーバは `HTTP 400` を返します。エンジンは生きて
  いて — 次のリクエストは普通に答えます — その文書について聞けない
  だけです。実用上の上限は、窓から生成の予算とチャットテンプレート分
  を引いた長さです。
- **長いプロンプトの最初のトークンは遅いです。** prefillは四段すべてで
  1620〜1641 tok/sに収まるので、194,544トークンのプロンプトは最初の
  トークンが出るまで120.04秒かかります。固まっているのではなく、
  prefill中です。`--enable-prefix-caching` を出荷フラグに入れたのは
  このためです: 長い文書を一本先頭に固定して設問だけ替えれば、初回の
  次からは桁が変わります — 130,990トークンで、初回79.86秒に対し
  キャッシュ後3.15秒。
- **約259,000トークンを超えると、このハードでは何も通りません。**
  `--max-model-len` を伸ばしても、`--gpu-memory-utilization` を
  下げても、`--max-num-batched-tokens` を縮めてもです: ホストの
  メモリが尽き、ノードのOOM狩りがvLLMのworkerを持っていきます。
  vLLMが起動時に印字する `peak activation` は8192トークンのダミー
  実行で測った値で、長文prefillで疎attentionのindexerが要求する
  一時領域は `--gpu-memory-utilization` の予算の外にあります。
  利用率を0.88から0.85へ下げても、死ぬまでがprefill開始62秒から
  155秒へ延びただけで、プロンプトは通りませんでした。

`RAY_memory_usage_threshold=0.99` は `serve/start-head.sh` と
`serve/start-worker.sh` が両ノードでexportしていて、これは省略でき
ません。ユニファイドメモリでは `--gpu-memory-utilization` がホスト
RAMから取られるため、Ray自身のOOMモニタはノードが既定の0.95を越えた
と見て、見つけられる最大のアクター、つまりvLLMのTP0 workerを殺します。
見えるのは `EngineDeadError` だけで — 起動中にも、リクエストの途中にも
出ます — vLLMのログには何も異常がなく、kill行はraylet側のログに
あります。二つのスクリプトはそれぞれ別のrayletを起こし、モニタも
raylet ごとに持つので、この変数は両方でexportする必要があります。

## うまくいかなかったものと時期

同じルーラーで測定したものです。探索空間を記録に残すために並べます。

<!-- failures:start -->

| item | what | cause | date |
|---|---|---|---|
| **moe-dsl-kernel-overlay** | Three successive failures. First, the patched kernel path was dead code: the backend is gated to device family 100 and never instantiates on sm_121, so an apparent -4 ms TPOT delta was run-to-run noise. Second, after re-hooking the class the deployment selects, every call failed a static eligibility check (the model's SwiGLU clamp limit) and silently fell back to stock: 256 calls, zero dispatches. Third, a host-side cute.make_layout call raised a TypeError at first use. | Instrumented, then fixed or abandoned each time; the overlay is not in the released config. | 2026-09-15 |
| **fp8-dense** | FP8-quantized dense linear variants ran slower than W4A16 NVFP4 on this stack (pair 1, sockets, 64-prompt ruler): no draft 9.3 tok/s / TPOT 107.1 ms against W4A16's 18.98 / 51.5; with the MTP K=2 draft 16.49 tok/s / TPOT 59.7 ms against W4A16's 24.01 / 40.7. | Measured slower; not adopted. | 2026-09-15 |
| **mtp-k-ge-3** | K=3 gave 19.39 tok/s and K=4 gave 17.44 tok/s vs K=2's 19.83 on an earlier requant route (sockets, pair 1); on stock (pair 2, sockets, expert parallel on), K=4 gave 15.2 and K=5 gave 13.1. | Per-position acceptance drops more quickly than the extra draft tokens save steps. K=2 was the best of the K values measured on this route; on route h + RDMA a first sweep-A pass measured K=1 at 34.28 and K=3 at 31.98 tok/s; the reverse-order pass was run and failed to boot at every K including K=0 (only 99.88 GiB was free on the device at startup against the 104.65 GiB the utilization target asks for), which is a startup-memory limit rather than a K effect, so the full same-checkpoint sweep in both K orders has still not produced a complete set of numbers. | 2026-09-16 |
| **ep-off** | Removing --enable-expert-parallel gained ~0.7% (pair 2, sockets, MTP K=2, 64-prompt ruler: 17.71 -> 17.84 tok/s). | Superseded. That reading was pair 2 over sockets at a 16384 window. On RDMA at 204800 the same removal is worth 6.5% (35.05 -> 37.33 tok/s on the same script) and lowers the post-READY high-water mark from 6014 to 4691 MiB, so the scripts now ship expert parallel off. | 2026-09-15 |
| **eager-mode** | --enforce-eager speeds up the stock single-stream step by 7.0 ms on the 256-step attribution totals (86.95 vs 93.98 ms) and by 13.4 ms on the C=1 TPOT median (77.5 vs 90.9 ms), but the gain mostly does not carry into MTP decode. | The step it speeds up is not the step speculation runs. Not adopted. | 2026-09-15 |
| **k4-k5-boot-oom** | On the STOCK checkpoint at these memory settings (128 GB unified memory per node, max seq len 16384, gpu-memory-utilization 0.86) the engine refused to start at K=5 twice: after the speculator's CUDA-graph capture (103 s and 3.03 GiB on one rank) the KV cache left was smaller than one max-seq-len request needs (0.89 GiB available against 0.92 GiB needed). One K=4 attempt died at the same capture step. Restricting the CUDA-graph capture sizes to the two sizes the C=1 measurement needs made both boot (READY 879 s and 901 s) and produced the K=4 15.2 and K=5 13.1 rows above, so this is a capture-size configuration limit, not a hard K ceiling. On route h + RDMA, K=1, K=3 and K=4 all booted (READY 885 s, 906 s and 918 s); the K=4 arm was stopped by the operator before it measured, so route h has no K=4 number, and K=5 was never run on route h at all. | Speculator graph capture grows with K and is taken out of the KV budget: supported for stock K=5 by the capture size and the KV figures above, and worked around by capturing fewer sizes. Two unrelated events are kept out of this column: an earlier K=4 attempt was killed by an orphaned pipeline's cleanup, and a later reverse-order route-h sweep failed to boot at every K including K=0 because only 99.88 GiB was free on the device at startup against the 104.65 GiB the utilization target asks for. | 2026-09-15 |
| **requant-loader-write** | A fused-KDA write missing the NVFP4 global scale produced KeyError ...weight_scale_2 at load. | Writer bug; now covered by verify.py config and the kda-quant overlay. | 2026-09-15 |
| **nccl-ll-symm-ar** | NCCL low-latency envs and symmetric-memory all-reduce booted but returned empty response bodies or died under concurrency. | Broken output path; unsafe, not adopted. | 2026-09-15 |
| **ev-ordering** | The expensive drafter-training data chain (extraction, two generation passes, conversion; ~5 h of wall time on the 2-GPU pair, ~10 GPU-hours) was queued ahead of the cheap K sweep; the trained drafter's holdout top-1 came out ~0.08, and the sweep showed K=2 was already the best of the measured values. | Sequenced by pipeline momentum instead of expected information per GPU-hour. | 2026-09-15 |

<!-- failures:end -->

上のクリーンルーム実行からは、さらに七つの失敗が出ました。どれもモデル・
transport・チェックポイントではなくランブックとドライバのもので、空の
ディレクトリから指示どおりに辿ったときにだけ現れる類のものです。五つを
挙げると、ランブック自身のシェルが受け付けなかった `mktemp` のテンプレート
(末尾の `X` が三つ未満)、ドライバが書いたのに一度も読み込まれなかった
serve設定、イメージ内のヘルパーがrootとして作り、次の段が書き込めなく
なったディレクトリのファイル、操作ノードとサービングのheadが同じ機械で
ある場合(スクリプトは自分自身へSSHしようとしました)、そしてJSONの
引数を囲む引用符を食べてしまったシェルのクォートです。いずれもここで公開
したファイルの側で直してあり、実行の側では直していません。

上記および下記の測定の時系列です。

| 日付 | 節目 |
|---|---|
| 2026-09-13 | pair 2でのstockベースライン: C=1で10.75 tok/s |
| 2026-09-15 | stock + 公式MTP K=2: pair 1で19.05 tok/s、pair 2で17.84 tok/s。代替レバーの多くはこの期間に失敗 |
| 2026-09-16 | route h再量子化、RDMAイメージ、pair 1でヘッドライン35.09 tok/s |

## NVIDIA NVFP4チェックポイントを出発点にした理由

公式の `nvidia/GLM-5.3-Flash-NVFP4` リリースはNVIDIA自身のツール
チェーンで量子化されており、そのモデルカードは実質的に精度劣化の
無いことを示すBF16対NVFP4のベンチマーク数値を公開しています
(モデルカードのリビジョン `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`)。
GPQA Diamond 0.9217 -> 0.9211、SciCode 0.5621 -> 0.5769、MMMU Pro
0.7688 -> 0.7630、AA-LCR 0.7100 -> 0.7106、IFBench 0.6130 -> 0.6054、
Terminal-Bench 2.1は0.8258 -> 0.8315です。これが信頼できる土台に
なる理由です。私のroute hがやるのは、公式リリースがBF16のまま
残した層へ同じ4-bit処理を広げることだけで、トレードオフが見え
続けるようゲートの数値を速度の数値の隣に置いています。これらの
カードの数値は標準的な公開ベンチマークによるもので、このリポジトリ
自身の厳しい採点器で採点した上のeval-200の合計とは比較できません。

## v2に残っていること

これは私が未完了だと分かっているものの一覧であって、ロードマップでは
ありません。どれがどれだけの価値になるかは予測しません。測れたときに
数値を出します。

**速度、まだ開いているもの**

- 自作のNVFP4 MoEカーネル。サービング中のリクエストで実際に走ったことは
  一度もありません。三回の試みは、GB10のビルドが生成しないクラス、次に
  間違ったクラス、そして例外を投げて黙ってオーバーレイを無効にしたホスト側
  の呼び出しに当たりました。エンジンのログに生存確認の行が出るまでは、
  測るものがありません。
- all-reduce。私が一度だけ取ったステップ帰属(eager、stock、シングル
  ストリーム、sockets上)では、ノード間のpeer waitが約87 msのデコード
  ステップのうち20.8 msでした。RDMAの下で同じ帰属を取り直したことは
  なく——一度試みましたが起動時に落ちてステップのデータは何も出ませ
  んでした——そのためこの待ちが速いtransport上でどれだけ残るかは
  分かりません。RDMAの行がtok/sで高いのはパス全体としての実測であり、
  このバケットへの帰属ではありません。collectiveの回数そのものを
  減らす試みもしていません。
- 学習したドラフトモデル。チェックポイント自身のMTPヘッドは、提案した
  トークンのうち0.62が受理されます。このモデル自身の出力で学習した
  ドラフトならもっと受理され、周期がすべて短くなります。最初の試みはデータ不足
  です——学習レコード594件、レシピ目標の約3Mトークンに対して約301k
  トークン、holdoutのスロットあたり命中は0.03-0.08——測る価値が出る
  までに、およそ十倍のデータが要ります。
- 投機の深さ。K=1(34.28)とK=2(35.09)はこの実験の実行ごとのばらつきでは
  分離できず、K=3はより遅いです。よいドラフトがあれば、その最適点
  の位置は変わります。

**品質、まだ測っていないもの**

- 英語とほかの言語。ここにあるprobeはすべて日本語です。
- コードの正しさ。コードのプロンプトは速度の表には出てきますが、品質の表
  には出てきません。
- 複数ターンの会話、tool呼び出し、指示追従、長文脈での品質——針の
  プローブは194,544トークンまで届いて40問中40問ですが、あれは検索
  であり、品質の表は2kほどより長いプロンプトでは何も走っていません
  ——安全性の挙動、そしてthinkingを有効にしたモデル。この
  族が普段使われる形です。
- より大きなコーパスでのperplexityと、その信頼区間。いまのprobeは八つの
  文で、コードはトークンごとの値を保持しないので、比に区間が付いていません。
- eval-200の対応のある検定。採点器は項目ごとの結果を捨てるので、同じ
  150項目を、本来そうであるはずの対として検定できません。

**再現、まだ完了していないもの**

- クリーンルームの実行は、ダウンロードと再量子化そのものを飛ばして
  います。重みはすでにノード上にありました。空のディスクから約204 GBの
  ダウンロードと約30分の再量子化を通す完全な経路は、まだ誰も端から端
  まで走らせていません。
- その実行は10プロンプトで、参照値の出どころである64ではありません。

**すでに効かないと分かっているもの。** dense linearへのFP8、散文での
K=3以上、そしてeagerモードの強制です。その数値と、測定したときの条件は
上の失敗表にあります。どれかを試し直す前に見る場所です。

## 終わらない部分

モデルを速くするのには一日の実験で足りました。その結果を公開できる形に
するほうが長くかかり、私が身構えていなかったのはそちらです。

足した検査の層はどれも何かを見つけ、見つかったものはどれも本物でした。
言い回しと漏れの機械検査は一回目で通ったので、本文のすべての数値をログ
まで辿る検査を足したところ、十四個が辿れませんでした。それを直し、次に
前提を知らない読み手に全体を読んでもらうと、ヘッドラインの比較が、別の
ハードウェアで、別のtransportで、別のプロンプト本数で測った二つの数値を
割っていたと分かりました。それを直し、次に同梱した証拠を表と突き合わせる
と、自分のログの別の行から来たスループットの数値が見つかりました。それを
直し、次に量子化スクリプトを本文の主張と照らして監査すると、間違った
チェックポイントを指されたら黙って壊れたドラフトを作るヘルパーが見つかり
ました。それを直し、次に品質の主張を敵対的に読んでもらい、四つの
カテゴリのうち一つの採点器がモデルの言い逃れを褒めることを知りました
——私の数値が動いたのは、まさにその向きでした。

どれも不注意ではありませんでした。それぞれに違う種類の見方が要りました。
そして型はいつも同じでした。検査を足し、何かを見つけ、直す。直しは小さい。
費用がかかるのは検査ではなく、層がいつももう一つあると気づくことのほう
です。

どこかで、作業が終わったからではなく、次の層が公開より価値が低いから止め
る、というのが誠実な一手になります。私はここで止めました。測っていないと
分かっているものは上の専用の節に並べてあり、検査しようと思いつかなかった
ものがあることこそ、これが最終版ではなくv1である理由です。見つけたら、
それは仕組みが働いたということです。聞けないより聞けたほうがありがたい。

## 名前について

侘び(わび)は、不完全なものや素朴なものを受け入れ、その中に豊かさを
見出す日本の感覚です。このリリースは発展途中のものをそのまま公開する
もので、改良は実測できたものから取り込みます。

## 次にやること

種類別の残りの集合、そして両方向のK順での完全な
K掃引です。ある構成がこのルーラーで35.09 tok/sを上回り、同じゲートを
通れば、v2として出します。

## ファイル

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
                            + check-headroom.sh + serve.env.example
scripts/agent-run.sh        one-shot driver for the whole runbook
scripts/verify-result.py    checks a measured row against the expected one
bench/                      measure.py + prompts-64.jsonl + eval-200.jsonl
                            + README.md (what the sets are)
results/                    results.tsv, quality.tsv, failures.tsv and the
                            sanitised per-run logs the tables cite
tests/                      run-tests.sh (offline smoke) +
                            check-md-invariants.py (wording-only edit check)
```

`eval-200.jsonl` のプロンプト二件は公開用に引用符を打ち直しました
(ASCII安全な引用 / 指示文の言い換え)。期待回答と採点は変わって
いません。`bench/prompts-64.jsonl` は測定したそのままの形で同梱して
います——[bench/README.md](bench/README.md)を参照してください。

## 誰が何をしたか

このリポジトリにある設計判断、検収基準、そしてすべての測定は私の
ものです。実装は、下に挙げたAIの席と
[CONTRIBUTORS.md](CONTRIBUTORS.md)に記載の席で進めました。ゲート・
実行・数値は、リリース前に私が確認しています。

## Contributors

- tenhkspark(メンテナ) — 心配し、見守り、出すと言った人。
- Claude Fable 5.1 (Anthropic) — 方向付け、実験設計、レビューと検収。
- Claude Opus 5 (Anthropic) — リリース前レビュー。
- Astra (OpenAI) — 測定と計画への敵対的レビュー。
- Devin SWE-2 (Cognition) — 実装、診断、実験キューの実行、
  リポジトリのドラフト作成。
- GLM-5.3 (Z.ai) — 実装担当。
- GLM-5.3-Flash (Z.ai) — 実装担当と要約。

## License

このリポジトリのコードはApache-2.0です。LICENSEを参照してください。
重みは別の話です: このリポジトリには含まれておらず、Hugging Faceで
公開している派生チェックポイントはApache-2.0ではなく、
`zai-org/GLM-5.3-Flash` から受け継いだ上流のMITライセンスに従います。
そのHugging FaceリポジトリにはMITの原文を逐語で同梱し、上流モデル・
NVIDIA Model Optimizerによる基底量子化・寄与者を記したNOTICEを
添えています。自分で重みを作る場合は、`requant/requant.py` で
NVIDIAチェックポイントを再量子化してください。
