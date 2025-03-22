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

from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner
from .vanilla_knn_mt import VanillaKNNMT
#from knnbox.models.vanilla_knn_mt import VanillaKNNMT, VanillaKNNMTDecoder

### Added for measuring KNN time
from fairseq.logging.meters import StopwatchMeter
import torch
import torch.nn as nn

# IT        0
# Koran     1
# Law       2
# Medical   3

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
        parser.add_argument("--knn-lambda", type=float, metavar="D", default=0.7,
                            help="The hyper-parameter lambda of vanilla knn-mt")
        parser.add_argument("--knn-temperature", type=float, metavar="D", default=10,
                            help="The hyper-parameter temperature of vanilla knn-mt")
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
        
class ExportsRouter(nn.Module):
    pass
        
class Expert(nn.Module):
    def __init__(self):
        pass
        
class MOELayer(nn.Module):
    def __init__(self, n_experts):
        self.experts = []


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
        
        # if args.knn_mode == "inference":
        #     # when inference, we don't load the keys, use its faiss index is enough
        #     self.datastore = Datastore.load(args.knn_datastore_path, load_list=["vals"])
        #     self.datastore.load_faiss_index("keys")
        #     self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
        #     self.combiner = Combiner(lambda_=args.knn_lambda,
        #              temperature=args.knn_temperature, probability_dim=len(dictionary))
            
        #     self.retrieve_timer = StopwatchMeter()

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
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
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if self.args.knn_mode == "build_datastore":
            raise RuntimeError("Can't build data store, please use moe KNN-MT to build datastore")
            
        elif self.args.knn_mode == "inference":
            ## query with x (x needn't to be half precision), 
            ## save retrieved `vals` and `distances`
            #record_timer_start(self.retrieve_timer)
            self.retriever.retrieve(x, return_list=["vals", "distances"])
            #record_timer_end(self.retrieve_timer)
        
        if not features_only:
            x = self.output_layer(x)
        return x, extra
    

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
        if self.args.knn_mode == "inference":
            #record_timer_start(self.retrieve_timer)
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            #record_timer_end(self.retrieve_timer)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)
        
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
    