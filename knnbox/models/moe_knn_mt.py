from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner
from .vanilla_knn_mt import VanillaKNNMT
from .less_retrieve_knn_mt import LessRetrieveSelectorSimple
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
### Added for measuring KNN time
from fairseq.logging.meters import StopwatchMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import itertools
torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

# IT        0
# Koran     1
# Law       2
# Medical   3
# Other     4
        
class ExpertsRouter(nn.Module):
    def __init__(self, embed_dim : int, n_experts : int):
        super().__init__()
        self.soft_router = nn.Linear(embed_dim, n_experts)
        self.noise_layer = nn.Linear(embed_dim, n_experts)
        
    def forward(self, X):
        return self.soft_router(X), self.noise_layer(X)
        
class Expert(nn.Module):
    def __init__(self, embed_dim, expert_hidden_dim = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, X):
        return self.net(X)
        
class MOELayer(nn.Module):
    def __init__(self, embed_dim, experts, domain_weight_if_domain_given = 0.5, residual_threshold = 0.5):
        super().__init__()
        self.n_experts = len(experts)
        self.router = ExpertsRouter(embed_dim, self.n_experts)
        if isinstance(experts, nn.Module):
            self.experts = experts
        elif isinstance(experts, list):
            self.experts = nn.ModuleList(experts)
        self.domain_weight_if_domain_given = domain_weight_if_domain_given
        self.residual_threshold = residual_threshold

    def forward(self, X):
        inputs = X["embedding"]  # [batch_size, seq_len, hidden_dim]
        logits = X['logits']     # maybe list of [batch_size, seq_len, vocab_size], len = n_experts
        if isinstance(logits, list):
            logits = torch.stack(logits)
        
        batch_size, seq_len, hidden_dim = inputs.shape
        vocab_size = logits[0].shape[-1]
        
        route_logits, noise_logits = self.router(inputs)   # [batch_size, seq_len, n_experts]
        
        # add noise
        noise = torch.randn_like(route_logits) * F.softplus(noise_logits)
        route_logits = route_logits + noise
        route_prob = F.softmax(route_logits, dim=-1)
        
        if "domain_label" in X and (not X["domain_label"] is None):
            domain = X['domain_label']  
            if domain.dim() == 1:
                domain = domain.unsqueeze(1).expand(-1, seq_len)
            domain_expert_mask = F.one_hot(domain, num_classes=self.n_experts).float()
            domain_prob = (route_prob * domain_expert_mask).sum(dim=-1, keepdim=True)
            residual_prob = route_prob * (1 - domain_expert_mask)
            residual_prob = residual_prob / (residual_prob.sum(dim=-1, keepdim=True) + 1e-6)
            residual_prob = residual_prob * (1 - self.domain_weight_if_domain_given)
            
            domain_prob = torch.full_like(domain_prob, self.domain_weight_if_domain_given)
            route_prob_revised = residual_prob + domain_expert_mask * domain_prob
            
            threshold = self.domain_weight_if_domain_given + self.residual_threshold 
        else:
            threshold = self.residual_threshold
            route_prob_revised = route_prob
        
        sorted_prob, sorted_idx = torch.sort(route_prob_revised, dim=-1, descending=True)
        cumulative_prob = torch.cumsum(sorted_prob, dim=-1) 
        mask = cumulative_prob < threshold
        topk_mask = mask.clone()
        topk_mask[..., 0] = True     # 至少一个专家
        topk_num = topk_mask.sum(dim=-1)
        max_k = topk_num.max().item()
        topk_indices = sorted_idx[..., :max_k]
        topk_weights = sorted_prob[..., :max_k]
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        #topk_weights = topk_weights * threshold    # [batch_size, seq_len, max_k]
        
        logits_p = logits.permute(1, 2, 0, 3).contiguous()   # [batch_size, seq_len, n_experts, vocab_size]
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, logits_p.size(-1)) 
        topk_logits = torch.gather(logits_p, dim=2, index=topk_indices_exp) 
        topk_weights_exp = topk_weights.unsqueeze(-1)
        combined_logits = (topk_logits * topk_weights_exp).sum(dim=2)
        return combined_logits, route_prob, route_prob_revised
    
        # 下面的是一个Naive的实现，不如上面的代码能充分利用tensor operation带来的GPU并行能力
        #combined_logits = torch.zeros(batch_size * seq_len, vocab_size, device=inputs.device)
        
        # for k in range(max_k):
        #     expert_id = topk_indices[..., k].view(-1)  
        #     weight_k = topk_weights[..., k].reshape(-1,1) 
        #     weight = weight_k.expand(-1, vocab_size)
           
        #     for i in range(self.n_experts):
        #         mask_i = (expert_id == i)
        #         if mask_i.any():
        #             indices = mask_i.nonzero(as_tuple=False).squeeze(-1)
        #             logits_i = logits[i].view(-1, vocab_size)
        #             logits_i = logits_i[indices]
        #             assert indices.max().item() < weight.size(0) and indices.min().item() >= 0
        #             weight_i = weight[indices]
        #             update_i = logits_i * weight_i
        #             combined_logits.index_add_(0, indices, update_i)
              
        #return combined_logits.view(batch_size, seq_len, vocab_size)
        
    
class LanguagePairWithDomainDataset(LanguagePairDataset):
    def __init__(self, domains, src, src_sizes, src_dict, tgt=None, tgt_sizes=None, tgt_dict=None, left_pad_source=True, left_pad_target=False, shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False, align_dataset=None, constraints=None, append_bos=False, eos=None, num_buckets=0, src_lang_id=None, tgt_lang_id=None, pad_to_multiple=1):
        super().__init__(src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, left_pad_source, left_pad_target, shuffle, input_feeding, remove_eos_from_source, append_eos_to_target, align_dataset, constraints, append_bos, eos, num_buckets, src_lang_id, tgt_lang_id, pad_to_multiple)
        self.domains = domains
        self._domains_tensor = torch.tensor(self.domains, dtype=torch.long)
        
    def collater(self, samples, pad_to_length=None):
        data = LanguagePairDataset.collater(self, samples, pad_to_length)
        data['domains'] = torch.stack([d['domain'] for d in samples])
        return data
    
    def __getitem__(self, idx):
        sample = LanguagePairDataset.__getitem__(self, idx)
        sample['domain'] = self._domains_tensor
        return sample

def load_langpair_with_domain_dataset(
    data_path,
    split,
    domain,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairWithDomainDataset(
        #torch.full((src_dataset.sizes.shape[0],), domain),
        domain,
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )
    

@register_task("translation_with_moe")
class TranslationTaskWithMOE(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
         
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)
    
    def load_dataset(self, split, combine=False, **kwargs):
        if 'epoch' in kwargs:    # To fix fairseq bugs
            epoch = kwargs['epoch']
            del kwargs['epoch']
        else:
            epoch = 1
        
        # paths = utils.split_paths(self.args.data)
        # assert len(paths) > 0
        # if split != getattr(self.args, "train_subset", None):
        #     # if not training data set, use the first shard for valid and test
        #     paths = paths[:1]
        # data_path = paths[(epoch - 1) % len(paths)]
        src, tgt = self.args.source_lang, self.args.target_lang
        if self.args.knn_mode == 'train':
            # Concat multi-domain dataset

            #base_dir = os.path.sep.join( os.path.split(os.path.dirname(self.args.data)) )
            base_dir = self.args.data
            dm_ds = []
            dm_name = ['it', 'koran', 'law', 'medical']
            for dm_id in range(0,4):
                data_path = os.path.join(base_dir, dm_name[dm_id])
                d = load_langpair_with_domain_dataset(
                    data_path,
                    split,
                    dm_id,
                    src,
                    self.src_dict,
                    tgt,
                    self.tgt_dict,
                    combine=combine,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    load_alignments=self.args.load_alignments,
                    truncate_source=self.args.truncate_source,
                    num_buckets=self.args.num_batch_buckets,
                    shuffle=(split != "test"),
                    pad_to_multiple=self.args.required_seq_len_multiple,
                )
                dm_ds.append(d)
            cd = ConcatDataset(dm_ds, sample_ratios=[1] * len(dm_ds))
            self.datasets[split] = cd
        elif self.args.knn_mode == 'inference':
            paths = utils.split_paths(self.args.data)
            assert len(paths) > 0
            if split != getattr(self.args, "train_subset", None):
                # if not training data set, use the first shard for valid and test
                paths = paths[:1]
            data_path = paths[(epoch - 1) % len(paths)]
            self.datasets[split] = load_langpair_with_domain_dataset(
                    data_path,
                    split,
                    0,
                    src,
                    self.src_dict,
                    tgt,
                    self.tgt_dict,
                    combine=combine,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    load_alignments=self.args.load_alignments,
                    truncate_source=self.args.truncate_source,
                    num_buckets=self.args.num_batch_buckets,
                    shuffle=(split != "test"),
                    pad_to_multiple=self.args.required_seq_len_multiple,
                )
        
@register_model("moe_knn_mt")
class MOEKNNMT(VanillaKNNMT):
    r"""
    The moe knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add pck knn-mt related args here
        """
        TransformerModel.add_args(parser=parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "inference", "train"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter k of vanilla knn-mt")
        # parser.add_argument("--knn-lambda", type=float, metavar="D", default=0.7,
        #                     help="The hyper-parameter lambda of vanilla knn-mt")
        # parser.add_argument("--knn-temperature", type=float, metavar="D", default=10,
        #                     help="The hyper-parameter temperature of vanilla knn-mt")
        parser.add_argument("--inference-router-threshold", type=float, default=0.5)
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with PckKNNMTDecoder
        """
        return MOEKNNMTDecoder(
            args,
            tgt_dict, 
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
        
    def forward(
        self,
        src_tokens,
        src_lengths,
        domain,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        target = None
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
    
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            domain=domain,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out
    
    def after_train_hook(self):
        self.decoder.after_train_hook()
        
class MOEKNNMTDecoder( TransformerDecoder ):
    r"""
    The moe knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        if args.knn_mode == "build_datastore":
            raise RuntimeError("Can't build data store, please use moe KNN-MT to build datastore")
            
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
       # self.base_dir = os.path.sep.join( os.path.split(os.path.dirname(self.args.data)) )
        self.base_dir = os.path.sep.join(os.path.dirname(__file__).split(os.path.sep)[:-2])
        lambdas = [0.7, 0.8, 0.8, 0.8]
        temps = [10, 100, 10, 10]
        self.datastore = []
        self.retriever = []
        self.combiner = []
          
        datastore_dir = os.path.join(self.base_dir,"datastore/vanilla")
        for i,dm in enumerate(["it", "koran", "law", "medical"]):
            ds_path = os.path.join(datastore_dir, dm)
            if not os.path.exists(ds_path):
                raise RuntimeError(f"Datastore for {dm} does not exist")
            ds = Datastore.load(ds_path, load_list=["vals"])
            ds.load_faiss_index("keys")
            rt = Retriever(datastore=ds, k=args.knn_k)
            self.datastore.append(ds)
            self.retriever.append(rt)
            combiner = Combiner(lambda_=lambdas[i],
                        temperature=temps[i], probability_dim=len(dictionary))
            self.combiner.append(combiner)
            
        if self.args.knn_mode == 'train':
            experts = []
            for i,dm in enumerate(["it", "koran", "law", "medical"]):
                rt_path = os.path.join(self.base_dir, f"save-models/LRKNNMT/{dm}/selector.pt")
                if not os.path.exists(rt_path):
                    raise RuntimeError(f"LR selector for `{dm}` does not exist, loaad from {rt_path}")
                
                selector : LessRetrieveSelectorSimple = torch.load(rt_path, map_location='cpu')
                expert = Expert(selector.l1.in_features, selector.l1.out_features)
                #expert.linear = selector.l1 # Not use now
                experts.append(expert)
        
            general_expert = Expert(selector.l1.in_features, selector.l1.out_features)
            experts.append(general_expert)
            
            self.moe = MOELayer( self.output_projection.in_features, experts )
        else:
            self.moe = torch.load(os.path.join(self.base_dir,"save-models/moe/moe.pt"), map_location='cpu')
            self.moe.residual_threshold = self.args.inference_router_threshold
          

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        domain = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        r"""
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        z, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if self.args.knn_mode == "build_datastore":
            raise RuntimeError("Can't build data store, please use moe KNN-MT to build datastore")
                        
        for rt_i in range(4):
            self.retriever[rt_i].retrieve(z, return_list=["vals", "distances"])
        
        x = self.output_layer(z)
        return x, z, domain, extra

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == "inference" or self.args.knn_mode == 'train':
            #record_timer_start(self.retrieve_timer)
            #knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            #combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            #record_timer_end(self.retrieve_timer)
            origin_logits = net_output[0]
            expert_logits = []
            for i in range(4):
                knn_prob = self.combiner[i].get_knn_prob(**self.retriever[i].results, device=net_output[0].device)
                knn_mt_logits, _ = self.combiner[i].get_combined_prob(knn_prob, net_output[0], log_probs=True)
                expert_logits.append(knn_mt_logits)

            if self.args.knn_mode == 'train':
                combined_prob, route_prob, route_prob_revised = self.moe(
                    {
                        "embedding" : net_output[1],
                        "domain_label" : net_output[2],
                        "logits" : (expert_logits + [origin_logits])
                    }   
                )
                self.last_route_prob = route_prob
                self.last_route_prob_revised = route_prob_revised
            else:
                combined_prob, _, _ = self.moe(
                    {
                        "embedding" : net_output[1],
                        "domain_label" : None,
                        "logits" : (expert_logits + [origin_logits])
                    }   
                )
            
            if log_probs:
                res = torch.log_softmax(combined_prob, dim=-1)
            else:
                res = torch.softmax(combined_prob, dim=-1)
            
            return res
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)
    
    def after_train_hook(self):
        os.makedirs(os.path.join(self.base_dir,"save-models/moe"),mode=0o775,exist_ok=True)
        torch.save(self.moe, os.path.join(self.base_dir,"save-models/moe/moe.pt"))
        
@register_criterion("moe_criterion")
class MOECriterion(FairseqCriterion):
    def __init__(self, task : TranslationTaskWithMOE, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.args = task.args
        if self.args.with_router_loss:
            logger.info("With router loss: Yes")
        else:
            logger.info("With router loss: No")
            
    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument("--with-router-loss", type=lambda x : x.lower() in ['y','t', '1','o'], default=True)
        
    def forward(self, model : MOEKNNMT, sample, reduce=True):
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        
        logits, last_hidden, domain, extra = model(
            src_tokens=sample['net_input']["src_tokens"],
            src_lengths=sample['net_input']["src_lengths"],
            domain=sample['domains'],
            prev_output_tokens=sample['net_input']["prev_output_tokens"],
            target=sample['target']
        )
        net_output = (logits, last_hidden, domain, extra)
        
        token_logits = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
        
        lprobs = token_logits.view(-1, token_logits.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss_mt = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="mean")
        
        router_lprobs = torch.log(model.decoder.last_route_prob).view(-1, model.decoder.last_route_prob.shape[-1])
        
        domain_target = domain.unsqueeze(-1).expand(-1, sample['target'].shape[-1]).contiguous()
        domain_target[(sample['target'] == self.padding_idx)] = -1
        domain_target = domain_target.view(-1)
        loss_router = F.nll_loss(
            router_lprobs,
            domain_target,
            ignore_index=-1,
            reduction='mean'
        )
        
        loss = loss_mt + loss_router
        
        if self.args.knn_mode != 'train':
            loss = loss * torch.tensor(0, device=logits.device, dtype=logits.dtype)
            
        logging_output = {
            "loss": loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        
        
        return loss, sample_size, logging_output
    
     
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_all_batches = [log["loss"].detach() for log in logging_outputs]
        mean_loss = torch.stack(loss_all_batches).mean().item()
        sample_size = sum(log.get("sample_size", 1) for log in logging_outputs)
        
        metrics.log_scalar(
            "loss" , mean_loss, sample_size, round=4
        )
        
        
r""" Define some moe knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("moe_knn_mt", "moe_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("moe_knn_mt", "moe_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("moe_knn_mt", "moe_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("moe_knn_mt", "moe_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("moe_knn_mt", "moe_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("moe_knn_mt", "moe_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("moe_knn_mt", "moe_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("moe_knn_mt", "moe_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)

@register_model_architecture("moe_knn_mt", "moe_knn_mt@transformer_zh_en")
def transformer_zh_en(args):
    archs.transformer_zh_en(args)
    