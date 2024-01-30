# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.fairseq_encoder import EncoderOut
from torch import Tensor


class KNNSequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    # 放在gpu上
    def cuda(self):
        self.model.cuda()
        return self

    ## 前馈翻译模型，prifix_tokens+sample作为输入，指定bos_token为bos_token
    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    # 和forward函数类似，均用来调用_generate函数
    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)


    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        # 翻译一开始，先初始化一个increment_state
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )

        net_input = sample["net_input"]

        # add by knnbox >>>>>>>>>>>>>>
        knn_parameter = sample["knn_parameter"] # 实时传入knn参数
        save_knn_informations = sample["save_knn_informations"] # bool变量，是否保存knn翻译工程中的信息
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if "src_tokens" in net_input:
            # 取得src tokens和src length
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        # 拿到bsz和srclen
        bsz, src_len = src_tokens.size()[:2]
        # beamsize是指定的
        beam_size = self.beam_size

        # constraints, 普通场景下不用管
        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            # 设定max_len, 为max_len_a*src_len+self.max_len_b和self.model.max_decoder_positions()-1的更小值
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        # compute the encoder output for each beam
        # 由于encoder只需要前馈一次，这里直接得到encoder_outs
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores

        # new_order是[bsz, beam_size]的格式，以
        # [3,2]为例,[[1,1],[2,2],[3,3]]
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        # 这里将encoder_out复制了beam_size份
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        # 这里用来存scores, 大小为[bsz*beam_size, max_len+1]
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        # 这里用来存tokens，大小为[bsz*beam_size, max_ken+2]. 保存eos和bos
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        # 如果没有指定bos，则用eos来作为开头. bos并不是模型生成的，因此这里主动填入
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None
        knn_probs_record = None # add by knnbox
        neural_probs_record = None
        combined_probs_record = None
        query_point_record = None
        knn_neighbors_values_record = None
        knn_neighbors_keys_record = None
        knn_l2_distance_record = None
        knn_sentence_ids_record = None
        knn_token_positions_record = None
        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        # 用来标识某个beam是否已经结束，如果已经结束那就不要再送入网络了
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        # 用来保存最终的sentence list
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step
        
        # 判断某个输入句子是否已经处理完成，和cands_to_ignore的粒度不同
        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        # batch里还剩多少个句子没有结束
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        # 每一步cand_size数
        ## TODO: 这里为什么是2*beam_size没懂
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        # bbsz用于确定每个sentence的beam七点
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            # 保存原始的batch_idxs
            original_batch_idxs = sample["id"]
        else:
            # 如果没有，那就自行编号0-bsz
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        # 开始大循环！max_len+1次decoder forward前馈
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            # print(f'step: {step}')
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            # 前馈decoder, forward decoder会拿到probs和avg_attn_scores
            # 这里我们同时拿到knn_probs, 参考lprobs的处理取得knn_probs
            # knn_probs的大小为[bsz*beam_size, len(target_dictionary)]
            lprobs, avg_attn_scores, extra = self.model.forward_decoder( 
                tokens[:, : step + 1], # prev_tokens是前面步已经生成的
                encoder_outs, # 完整的encoder out
                incremental_states, # incremental_states
                self.temperature,
                knn_parameter,  # knn的参数
                save_knn_informations, # 标识是否存储knn的一些信息
                sample, # 本来不该传sample的，但是要传knn相关的一些东西
            )

            # addby knnbox >>>>>>>>>>>>>>>>>>>>>>>
            
            neural_probs = extra.get("neural_probs")
            combined_probs = extra.get("combined_probs")
            query_point = extra.get("query_point")
            knn_neighbors_values = extra.get("knn_neighbors_values")
            knn_neighbors_keys = extra.get("knn_neighbors_keys")
            knn_l2_distance = extra.get("knn_l2_distance")
            knn_sentence_ids = extra.get("knn_sentence_ids")
            knn_token_positions = extra.get("knn_token_positions")

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
            # 手动处理pad项的概率，将其置为0
            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            # 如果到了max_len,强制选eos
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                # 如果attn是空的，则为其分配空间
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                # 将该步的attn weight放到attn的step+1位置
                attn[:, :, step + 1].copy_(avg_attn_scores)
            

            # add by knnbox >>>>>>>>>>>>>>>>>>>>>>>
            if neural_probs is not None:
                if neural_probs_record is None:
                    neural_probs_record = torch.empty(
                        bsz*beam_size, max_len+2, len(self.tgt_dict)
                    ).to(scores)
                # 将prob放到对应步的位置上
                neural_probs_record[:, step+1, :] = neural_probs.squeeze(1)
            
            if combined_probs is not None:
                if combined_probs_record is None:
                    combined_probs_record = torch.empty(
                        bsz*beam_size, max_len+2, len(self.tgt_dict)
                    ).to(scores)
                # 将prob放到对应步的位置上
                combined_probs_record[:, step+1, :] = combined_probs.squeeze(1)

            if query_point is not None:
                if query_point_record is None:
                    query_point_record = torch.empty(
                        # TODO: 硬编码改掉
                        bsz*beam_size, max_len+2, query_point.shape[-1]
                    ).to(scores)
                # 将prob放到对应步的位置上
                query_point_record[:, step+1, :] = query_point.squeeze(1)
            
            if knn_neighbors_values is not None:
                if knn_neighbors_values_record is None:
                    knn_neighbors_values_record = torch.empty(
                        bsz*beam_size, max_len+2, int(sample["knn_parameter"]["k"]),
                        dtype=torch.int32,
                    ).to(scores.device)
                knn_neighbors_values_record[:, step+1, :] = knn_neighbors_values.squeeze(1)

            if knn_neighbors_keys is not None:
                if knn_neighbors_keys_record is None:
                    knn_neighbors_keys_record = torch.empty(
                        bsz*beam_size, max_len+2, sample["knn_parameter"]["k"], knn_neighbors_keys.shape[-1],
                    ).to(scores)
                knn_neighbors_keys_record[:, step+1, :] = knn_neighbors_keys.squeeze(1)

            if knn_l2_distance is not None:
                if knn_l2_distance_record is None:
                    knn_l2_distance_record = torch.empty(
                        bsz*beam_size, max_len+2, int(sample["knn_parameter"]["k"])
                    ).to(scores)
                
                knn_l2_distance_record[:, step+1, :] = knn_l2_distance.squeeze(1)


            if knn_sentence_ids is not None:
                if knn_sentence_ids_record is None:
                    knn_sentence_ids_record = torch.empty(
                        bsz*beam_size, max_len+2, int(sample["knn_parameter"]["k"]),
                        dtype=torch.int32,
                    ).to(scores.device)
                knn_sentence_ids_record[:, step+1, :] = knn_sentence_ids.squeeze(1)

            if knn_token_positions is not None:
                if knn_token_positions_record is None:
                    knn_token_positions_record = torch.empty(
                        bsz*beam_size, max_len+2, int(sample["knn_parameter"]["k"]),
                        dtype=torch.int32,
                    ).to(scores.device)
                knn_token_positions_record[:, step+1, :] = knn_token_positions.squeeze(1)

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            scores = scores.type_as(lprobs)
            # 存放finished indices和finished scores
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            # 此处将forward和beam search step分开
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            # bbsz_offset和cand_beams相加得到真正的cand_bbsz_idx
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                    neural_probs_record, # add by knnbox
                    combined_probs_record, # add by knnbox
                    query_point_record, # add by knnbox
                    knn_neighbors_keys_record,
                    knn_neighbors_values_record,
                    knn_l2_distance_record,
                    knn_sentence_ids_record,
                    knn_token_positions_record,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len
            
            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                # add by knnbox >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if neural_probs_record is not None:
                    neural_probs_record = neural_probs_record.view(bsz, -1)[batch_idxs].view(
                        new_bsz*beam_size, neural_probs_record.size(1), -1
                    )
                if combined_probs_record is not None:
                    combined_probs_record = combined_probs_record.view(bsz, -1)[batch_idxs].view(
                        new_bsz*beam_size, combined_probs_record.size(1), -1
                    )
                
                if query_point_record is not None:
                    query_point_record = query_point_record.view(bsz, -1)[batch_idxs].view(
                        new_bsz*beam_size, query_point_record.size(1), -1
                    )
                if knn_neighbors_keys_record is not None:
                    knn_neighbors_keys_record = knn_neighbors_keys_record.view(bsz, -1)[batch_idxs].view(
                        new_bsz*beam_size, knn_neighbors_keys_record.size(1), -1
                    )
                if knn_neighbors_values_record is not None:
                    knn_neighbors_values_record = knn_neighbors_values_record.view(bsz, -1)[batch_idxs].view(
                        new_bsz*beam_size, knn_neighbors_values_record.size(1), -1
                    )
                if knn_l2_distance_record is not None:
                    knn_l2_distance_record = knn_l2_distance_record.view(bsz, -1)[batch_idxs].view(
                        new_bsz*beam_size, knn_l2_distance_record.size(1), -1
                    )
                if knn_sentence_ids_record is not None:
                    knn_sentence_ids_record = knn_sentence_ids_record.view(bsz, -1)[batch_idxs].view(
                        new_bsz*beam_size, knn_sentence_ids_record.size(1),-1
                    )
                if knn_token_positions_record is not None:
                    knn_token_positions_record = knn_token_positions_record.view(bsz, -1)[batch_idxs].view(
                        new_bsz*beam_size, knn_token_positions_record.size(1), -1
                    )
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )
            
            # add by knnbox >>>>>>>>>>>>>>copy neural_prob for activate hypotheses
            if neural_probs_record is not None:
                neural_probs_record[:,:step+2,:] = torch.index_select(
                    neural_probs_record[:,:step+2,:], dim=0, index=active_bbsz_idx
            )

            if combined_probs_record is not None:
                combined_probs_record[:,:step+2,:] = torch.index_select(
                    combined_probs_record[:,:step+2,:], dim=0, index=active_bbsz_idx
                )
            if query_point_record is not None:
                query_point_record[:,:step+2,:] = torch.index_select(
                    query_point_record[:,:step+2,:], dim=0, index=active_bbsz_idx
                )
            if knn_neighbors_keys_record is not None:
                knn_neighbors_keys_record[:,:step+2,:] = torch.index_select(
                    knn_neighbors_keys_record[:,:step+2,:], dim=0, index=active_bbsz_idx
                )
            if knn_neighbors_values_record is not None:
                knn_neighbors_values_record[:,:step+2,:] = torch.index_select(
                    knn_neighbors_values_record[:,:step+2,:], dim=0, index=active_bbsz_idx
                )
            if knn_l2_distance_record is not None:
                knn_l2_distance_record[:,:step+2,:] = torch.index_select(
                    knn_l2_distance_record[:,:step+2,:], dim=0, index=active_bbsz_idx
                )
            
            if knn_sentence_ids_record is not None:
                knn_sentence_ids_record[:, :step+2, :] = torch.index_select(
                    knn_sentence_ids_record[:,:step+2,:], dim=0, index=active_bbsz_idx
                )
            if knn_token_positions_record is not None:
                knn_token_positions_record[:, :step+2,:] = torch.index_select(
                    knn_token_positions_record[:,:step+2,:], dim=0, index=active_bbsz_idx
                )
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
        neural_probs_record: Optional[Tensor], # add by knnbox
        combined_probs_record: Optional[Tensor], # add by knnbox
        query_point_record,
        knn_neighbors_keys_record,
        knn_neighbors_values_record,
        knn_l2_distance_record,
        knn_sentence_ids_record,
        knn_token_positions_record,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )
        # add by knnbox >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        neural_prob_record_clone = (
            neural_probs_record.index_select(0, bbsz_idx)[:,1:step+2,:]
            if neural_probs_record is not None
            else None
        )
        combined_prob_record_clone = (
            combined_probs_record.index_select(0, bbsz_idx)[:, 1:step+2,:]
            if combined_probs_record is not None
            else None
        )
        query_point_record_clone = (
            query_point_record.index_select(0, bbsz_idx)[:, 1:step+2,:]
            if query_point_record is not None
            else None
        )
        knn_neighbors_keys_record_clone = (
            knn_neighbors_keys_record.index_select(0, bbsz_idx)[:, 1:step+2,:]
            if knn_neighbors_keys_record is not None
            else None
        )
        knn_neighbors_values_record_clone = (
            knn_neighbors_values_record.index_select(0, bbsz_idx)[:, 1:step+2,:]
            if knn_neighbors_values_record is not None
            else None
        )
        knn_l2_distance_record_clone = (
            knn_l2_distance_record.index_select(0, bbsz_idx)[:, 1:step+2,:]
            if knn_l2_distance_record is not None
            else None
        )
        knn_sentence_ids_record_clone = (
            knn_sentence_ids_record.index_select(0, bbsz_idx)[:, 1:step+2,:]
            if knn_sentence_ids_record is not None
            else None
        )
        knn_token_positions_record_clone = (
            knn_token_positions_record.index_select(0, bbsz_idx)[:, 1:step+2,:]
            if knn_token_positions_record is not None
            else None
        )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # set() is not supported in script export

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # print(f"{step} FINISHED {idx} {score} {sent}={unfin_idx} {cum_unfin}")
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)
                
                # add by knnbox >>>>>>>>>>>>>>>>>
                if neural_prob_record_clone is not None:
                    hypo_neural_prob_record = neural_prob_record_clone[i]
                else:
                    hypo_neural_prob_record = torch.empty(0)

                if combined_prob_record_clone is not None:
                    hypo_combined_prob_record = combined_prob_record_clone[i]
                else:
                    hypo_combined_prob_record = torch.empty(0)

                if combined_prob_record_clone is not None:
                    hypo_combined_prob_record = combined_prob_record_clone[i]
                else:
                    hypo_combined_prob_record = torch.empty(0)
                
                if query_point_record_clone is not None:
                    hypo_query_point_record = query_point_record_clone[i]
                else:
                    hypo_query_point_record = torch.empty(0)
                
                if knn_neighbors_keys_record_clone is not None:
                    hypo_knn_neighbors_keys_record = knn_neighbors_keys_record_clone[i]
                else:
                    hypo_knn_neighbors_keys_record = torch.empty(0)

                if knn_neighbors_values_record_clone is not None:
                    hypo_knn_neighbors_values_record = knn_neighbors_values_record_clone[i]
                else:
                    hypo_knn_neighbors_values_record = torch.empty(0)

                if knn_l2_distance_record_clone is not None:
                    hypo_knn_l2_distance_record = knn_l2_distance_record_clone[i]
                else:
                    hypo_knn_l2_distance_record = torch.empty(0)

                if knn_sentence_ids_record_clone is not None:
                    hypo_knn_sentence_ids_record = knn_sentence_ids_record_clone[i]
                else:
                    hypo_knn_sentence_ids_record = torch.empty(0)
                
                if knn_token_positions_record_clone is not None:
                    hypo_knn_token_positions_record = knn_token_positions_record_clone[i]
                else:
                    hypo_knn_token_positions_record = torch.empty(0)
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "neural_probs": hypo_neural_prob_record,  # add by knnbox, [tgt_len, len(target_dict)]
                        "combined_probs": hypo_combined_prob_record, # add by knnbox
                        "query_point": hypo_query_point_record,
                        "knn_neighbors_keys": hypo_knn_neighbors_keys_record,
                        "knn_neighbors_values": hypo_knn_neighbors_values_record,
                        "knn_l2_distance": hypo_knn_l2_distance_record,
                        "knn_sentence_ids": hypo_knn_sentence_ids_record,
                        "knn_token_positions": hypo_knn_token_positions_record,
                    }
                )

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def calculate_banned_tokens(
        self,
        tokens,
        step: int,
        gen_ngrams: List[Dict[str, List[int]]],
        no_repeat_ngram_size: int,
        bbsz_idx: int,
    ):
        tokens_list: List[int] = tokens[
            bbsz_idx, step + 2 - no_repeat_ngram_size : step + 1
        ].tolist()
        # before decoding the next token, prevent decoding of ngrams that have already appeared
        ngram_index = ",".join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    def transpose_list(self, l: List[List[int]]):
        # GeneratorExp aren't supported in TS so ignoring the lint
        min_len = min([len(x) for x in l])  # noqa
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        # for each beam and batch sentence, generate a list of previous ngrams
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {})
            for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list(
                [gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
            ):
                key = ",".join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
                    key, torch.jit.annotate(List[int], [])
                ) + [ngram[-1]]

        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(
                    tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
                )
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
            ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx]).long()
            ] = torch.tensor(-math.inf).to(lprobs)
        return lprobs


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[EncoderOut],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
        knn_parameter = None,    # add by knnbox
        save_knn_informations = None,  # add by knnbox
        sample = None, # add by knnbox
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[EncoderOut] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                    knn_parameter=knn_parameter, # add by knnbox
                    save_knn_informations=save_knn_informations,    # add by knnbox
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out,
                knn_parameter=knn_parameter, save_knn_informations=save_knn_informations # add by knnbox
                )

            attn: Optional[Tensor] = None
            knn_probs: Optional[Tensor] = None # add by knnbox

            decoder_len = len(decoder_out)
            # 这里是取attn
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs, extra = model.get_normalized_probs(  # add by knnbox
                decoder_out_tuple, log_probs=True, sample=sample
            )


            
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn, extra   # add by knnbox, we simplely assume model size == 1

            
            log_probs.append(probs)
         
            
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )
        extra = {}
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn, extra # modified by knnbox, add extra 

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Optional[List[EncoderOut]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[EncoderOut] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )


class SequenceGeneratorWithAlignment(KNNSequenceGenerator):
    def __init__(self, models, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(EnsembleModelWithAlignment(models), tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        finalized = super()._generate(sample, **kwargs)

        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        (
            src_tokens,
            src_lengths,
            prev_output_tokens,
            tgt_tokens,
        ) = self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, "full_context_alignment", False) for m in self.model.models):
            attn = self.model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]["attention"].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        if src_tokens.device != "cpu":
            src_tokens = src_tokens.to("cpu")
            tgt_tokens = tgt_tokens.to("cpu")
            attn = [i.to("cpu") for i in attn]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(
                attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos
            )
            finalized[i // beam_size][i % beam_size]["alignment"] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        src_tokens = (
            src_tokens[:, None, :]
            .expand(-1, self.beam_size, -1)
            .contiguous()
            .view(bsz * self.beam_size, -1)
        )
        src_lengths = sample["net_input"]["src_lengths"]
        src_lengths = (
            src_lengths[:, None]
            .expand(-1, self.beam_size)
            .contiguous()
            .view(bsz * self.beam_size)
        )
        prev_output_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]["attn"][0]
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn
