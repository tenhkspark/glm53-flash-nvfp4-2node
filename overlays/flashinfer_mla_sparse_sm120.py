# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]

        # Native no-rope MLA (qk_rope_head_dim == 0, e.g. GLM-5.3-Flash):
        # the SM120 GLM-NSA kernel requires a d_qk=576 layout, so this stores the
        # 512-wide latent KV in the fp8_ds_mla blob with the 64 rope slots
        # zero-filled, and pad the (latent) query with 64 zero rope dims. The
        # zero rope dims contribute exactly 0 to every dot product, so the
        # 576-wide kernel computes the same attention as a native no-rope one.
        self.is_nope_mla = self.qk_rope_head_dim == 0
        if self.is_nope_mla and (
            self.qk_nope_head_dim != 256 or self.kv_lora_rank != 512
        ):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 no-rope support requires "
                "qk_nope_head_dim=256 and kv_lora_rank=512, got "
                f"qk_nope_head_dim={self.qk_nope_head_dim}, "
                f"kv_lora_rank={self.kv_lora_rank}"
            )

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # kpool (key-pooling) indexer: the top-k buffer reserves (kpool - 1)
        # columns after index_topk for the incomplete trailing pool ("tail"),
        # then rounds the whole width up to a multiple of 128 (see
        # models/glm5next/nvidia/model.py). The SM120 GLM kernel requires the
        # page table to be exactly index_topk wide, so forward_mqa folds the
        # tail columns into the last top-k slots before truncating.
        hf_text_config = (
            vllm_config.model_config.hf_text_config
            if vllm_config.model_config is not None
            else None
        )
        kpool = getattr(hf_text_config, "index_kpool", None)
        self.kpool_tail_width: int = (int(kpool) - 1) if kpool and int(kpool) > 1 else 0

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        # Same as MLAAttentionImpl.do_kv_cache_update, but for no-rope models
        # the incoming k_pe has 0 width while concat_and_cache_mla's
        # fp8_ds_mla path requires pe_dim=64: fill the rope slots with zeros.
        if kv_cache.numel() == 0:
            return
        if self.is_nope_mla and k_pe.shape[-1] == 0:
            k_pe = k_pe.new_zeros((*k_pe.shape[:-1], 64))
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        # No-rope: pad the latent query (kv_lora_rank wide) to the d_qk=576
        # layout the SM120 GLM-NSA kernel expects; rope dims are zeros (see
        # __init__). Kernels with a real rope pass through unchanged.
        qk_rope_head_dim_arg = self.qk_rope_head_dim
        if self.is_nope_mla:
            q = torch.nn.functional.pad(q, (0, 64))
            qk_rope_head_dim_arg = 64

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        topk_indices_physical = cast(
            torch.Tensor,
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            ),
        )

        # The top-k buffer is wider than index_topk (kpool tail + 128-rounding).
        # The kernel wants exactly index_topk columns: move the tail entries
        # into the last top-k slots (dropping that many lowest-ranked indexed
        # tokens) and truncate. Padding columns beyond the tail are -1 and
        # stay masked by the kernel.
        sparse_topk_capacity = attn_metadata.topk_tokens
        if topk_indices_physical.shape[1] != sparse_topk_capacity:
            tail_w = self.kpool_tail_width
            if tail_w:
                topk_indices_physical[
                    :, sparse_topk_capacity - tail_w : sparse_topk_capacity
                ] = topk_indices_physical[
                    :, sparse_topk_capacity : sparse_topk_capacity + tail_w
                ]
            topk_indices_physical = topk_indices_physical[
                :, :sparse_topk_capacity
            ].contiguous()

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim_arg,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=attn_metadata.topk_tokens,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            kv_scale_format=self.kv_scale_format,
        )
        return out.squeeze(1), None
