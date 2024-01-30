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

import torch
import numpy as np

MODEL_NAME = "vanilla_knn_mt_inspect_redundant"

@register_model(MODEL_NAME)
class VanillaKNNMTInspectRedundant(TransformerModel):
    r"""
    The vanilla knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "inference"],
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
        we override this function, replace the TransformerDecoder with VanillaKNNMTDecoder
        """
        return VanillaKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
        
    def after_inference_hook(self):
        if hasattr(self.decoder, "after_inference_hook"):
            self.decoder.after_inference_hook()


class VanillaKNNMTDecoder(TransformerDecoder):
    r"""
    The vanilla knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]

        elif args.knn_mode == "inference":
            # when inference, we don't load the keys, use its faiss index is enough
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=["vals"])
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.combiner = Combiner(lambda_=args.knn_lambda,
                     temperature=args.knn_temperature, probability_dim=len(dictionary))
            
            self.num_redundant = 0
            self.num_total_samples = 0
            
            self.redundant_tids = np.zeros(shape=(embed_tokens.weight.shape[0],), dtype=int)
            self.redundant_vocabs = {}

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
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            # save half precision keys
            self.datastore["keys"].add(keys.half())
        
        elif self.args.knn_mode == "inference":
            ## query with x (x needn't to be half precision), 
            ## save retrieved `vals` and `distances`
            self.retriever.retrieve(x, return_list=["vals", "distances"])
        
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
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            
            batch_size = knn_prob.shape[0]
            
            redundant_label = (torch.argmax(torch.softmax(combined_prob,dim=-1),dim=-1) == torch.argmax(torch.softmax(net_output[0],dim=-1),dim=-1)).long()
            
            redundant_token_ids = (torch.argmax(torch.softmax(combined_prob,dim=-1),dim=-1)).view(-1)[redundant_label.view(-1)].detach().cpu().numpy()
            redundant_count = np.bincount(redundant_token_ids)
            
            for i, t in enumerate(redundant_count):
                self.redundant_tids[i] += t
            
            self.num_total_samples += batch_size
            self.num_redundant += redundant_label.sum()
                
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)
        
    def after_inference_hook(self):
        print(f"Redundant ratio = {self.num_redundant / self.num_total_samples} ({self.num_redundant}/{self.num_total_samples})")
        for i in range(self.redundant_tids.shape[0]):
            if self.redundant_tids[i] == 0:
                continue
            w = self.dictionary[i]
            self.redundant_vocabs[w] = self.redundant_vocabs.get(w, 0) + self.redundant_tids[i]
            
        del self.redundant_vocabs['</s>']
        s = sorted(zip(self.redundant_vocabs.values(), self.redundant_vocabs.keys()))
        print(s[-8:])


r""" Define some vanilla knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture(MODEL_NAME, f"{MODEL_NAME}@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture(MODEL_NAME, f"{MODEL_NAME}@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture(MODEL_NAME, f"{MODEL_NAME}@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(MODEL_NAME, f"{MODEL_NAME}@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture(MODEL_NAME, f"{MODEL_NAME}@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture(MODEL_NAME, f"{MODEL_NAME}@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture(MODEL_NAME, f"{MODEL_NAME}@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture(MODEL_NAME, f"{MODEL_NAME}@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)

@register_model_architecture(MODEL_NAME, f"{MODEL_NAME}@transformer_zh_en")
def transformer_zh_en(args):
    archs.transformer_zh_en(args)
