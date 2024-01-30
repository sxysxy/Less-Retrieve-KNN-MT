from typing import Any, Dict, Iterator, List, Optional, Tuple
from torch import Tensor
from torch.nn.parameter import Parameter
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
import torch.nn as nn
import torch.nn.functional as F
import os
import fairseq
from fairseq.criterions import register_criterion, FairseqCriterion
from torch.autograd import Variable
from .dice_loss import DiceLoss
from knnbox.common_utils import Memmap, read_config, write_config
from knnbox.datastore.utils import build_faiss_index, load_faiss_index 
import logging
from fairseq.logging.meters import StopwatchMeter
from .vanilla_knn_mt import record_timer_start, record_timer_end

import math

class LessRetrieveSelectorSimple(nn.Module):
    def __init__(self,input_size,
                 hidden_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        #self.l2 = nn.Linear(hidden_size, hidden_size)
        self.lo = nn.Linear(hidden_size, 2)
     

    def forward(self, x):
        x = F.relu(self.l1(x))
        #x = F.dropout(x, p=0.1)
        #x = F.relu(self.l2(x))
        #x = F.dropout(x, p=0.1)
        #x = torch.tanh(self.l1(x))
        return self.lo(x)
    
    def save(self, path):
        # Called from save_checkpoint function in fairseq/checkpoint_utils.py(modified)
        os.makedirs(os.path.dirname(path), mode=0o755, exist_ok=True)
        torch.save(self, path)
        
    @classmethod
    def load(cls, path):
        return torch.load(path)
    
class LessRetrieveSelector(nn.Module):
    def __init__(self,input_size,
                 hidden_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, input_size)
        #self.l3 = nn.Linear(hidden_size, 2)
        #self.li = nn.MultiheadAttention(input_size, 2, batch_first=True)
        self.lo = nn.Linear(input_size, 2)
     

    def forward(self, x):
        #x, _ = self.li(query=x,key=x,value=x)
        #x = F.relu(x)
        z = self.l2(self.l1(F.layer_norm(x, normalized_shape=x.shape))) + x
        return self.lo(F.relu(z))
    
    def save(self, path):
        # Called from save_checkpoint function in fairseq/checkpoint_utils.py(modified)
        os.makedirs(os.path.dirname(path), mode=0o755, exist_ok=True)
        torch.save(self, path)
        
    @classmethod
    def load(cls, path):
        return torch.load(path)


class GumbelSoftmaxLayer(nn.Module):
    def __init__(self, total_epochs, init_tau = 10, min_tau = 0.1, update_tau = None) -> None:
        super().__init__()
        self.init_tau = init_tau
        self.min_tau = min_tau
        
        def linear_update_tau(cur_epoch):
            return init_tau - (init_tau - min_tau) * cur_epoch / (total_epochs)
            
        self.update_tau = update_tau if update_tau else linear_update_tau
        
        self.reset_for_retrain()
        
        
    def forward(self, X):
       tau = self.update_tau(self.current_epoch)
       return F.gumbel_softmax(X, dim=-1, tau=tau, hard=False)
    
    
    def reset_for_retrain(self):
        self.current_epoch = 0

@register_criterion("less_retrieve_criterion")
class LessRetrieveCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
      #  self.loss_weight = torch.nn.Parameter(torch.ones(2, requires_grad=True))
        
        
    def forward(self, model, sample, reduce=True):
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        
        out, extra, selector_data = model(
            src_tokens=sample['net_input']["src_tokens"],
            src_lengths=sample['net_input']["src_lengths"],
            prev_output_tokens=sample['net_input']["prev_output_tokens"],
            target=sample['target']
        )
        
        if extra["args"].knn_mode != 'train_less_retrieve':
            a = torch.tensor(1.0)
            b = torch.tensor(2.0, requires_grad=True)
            loss = b-a
            return loss, sample_size, {"loss" : loss}  # Just make the fairseq framework happy
        
        selector_pred = selector_data["selector_pred"].view(-1,2)
        
        selector_label = selector_data["selector_label"].view(-1)
        is_padding = (sample["target"] == self.padding_idx).view(-1)
        selector_label[is_padding] = -100  #ignore padding idx
        
        # Dice loss always leads recall and retrieve_ratio = 1
        # dice = DiceLoss(with_logits=False, ohem_ratio=4)
        # loss_for_cls = dice(selector_pred, selector_label)

        pos_flag = (selector_label == 0)
        neg_flag = (selector_label == 1)
        #num_pos = pos_flag.int().sum()
        #num_neg = neg_flag.int().sum()
        
        #weight =  (num_neg + num_pos) / torch.tensor([num_pos, num_neg], dtype=torch.float, device=out.device)
        #weight /= weight.max()
        #weight = None
        #loss_for_cls = F.cross_entropy(selector_pred, selector_label, weight=weight, ignore_index=-100) 
        
        negative_weight = (selector_label == 1).int().sum() / selector_label.shape[0]
        loss_for_cls = F.cross_entropy(selector_pred, selector_label, weight=torch.tensor([negative_weight,(1-negative_weight)], dtype=torch.float, device=out.device))
        #cls_focal_loss = FocalLoss(2, alpha=torch.tensor([[negative_weight],[(1-negative_weight)]], dtype=torch.float, device=out.device))
        #loss_for_cls = cls_focal_loss(selector_pred, selector_label)
        
        selector_pred_max = F.softmax(selector_pred,dim=-1).argmax(dim=-1)
        #selector_pred_max[is_padding] = -100  #ignore padding idx
        pred_pos = (selector_pred_max == 0)
        pred_neg = (selector_pred_max == 1)
        
                
        # Let 0 to be positive, 1 to be negative
        tp = torch.logical_and(pred_pos, pos_flag).int().sum()
        fp = torch.logical_and(pred_pos, neg_flag).int().sum()
        tn = torch.logical_and(pred_neg, neg_flag).int().sum()
        fn = torch.logical_and(pred_neg, pos_flag).int().sum()
        
        num_pred_positive = (selector_pred_max == 0).int().sum()
        num_total_tokens = (selector_label).shape[0]
        
        if extra["args"].use_mt_loss_for_selector:
            
            lprobs = model.get_normalized_probs((out,{}), log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, (out,{})).view(-1)
            loss_for_mt = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="mean",
            )

            loss = self.weighted_loss(loss_for_cls, loss_for_mt)
            
        else:
            
            loss = loss_for_cls
        
        logging_output = {
            "loss": loss,
            "tp" : tp,
            "fp" : fp,
            "tn" : tn,
            "fn" : fn,
            "num_pred_positive" : num_pred_positive,
            "num_total_tokens" : num_total_tokens,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        
        return loss, sample_size, logging_output
    
    def weighted_loss(self, loss_for_cls, loss_for_mt):
        # Eq. 7
        # Due to reduce = 'mean', here we can directly add these two loss, no need for division by N
        return loss_for_cls + loss_for_mt 
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_all_batches = [log["loss"].item() for log in logging_outputs]
        ntokens = sum(log.get("num_total_tokens", 0) for log in logging_outputs)
        npredpos = sum(log.get("num_pred_positive", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        tp = sum(log.get('tp',0) for log in logging_outputs)
        fp = sum(log.get('fp',0) for log in logging_outputs)
        tn = sum(log.get('tn', 0) for log in logging_outputs)
        fn = sum(log.get('fn', 0) for log in logging_outputs)
        if (tp + fp + tn + fn) != 0:
            accuracy = (tp + tn) / (tp + fp + tn + fn)
        else:
            accuracy = 0
            
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0 
        
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        # we divide by log(2) to convert the loss from base e to base 2
        fairseq.metrics.log_scalar(
            "loss", sum(loss_all_batches) / len(loss_all_batches), sample_size, round=4
        )
        fairseq.metrics.log_scalar(
            "accuracy", accuracy, sample_size, round=4
        )
        fairseq.metrics.log_scalar(
            "precision", precision, sample_size, round=4
        )
        fairseq.metrics.log_scalar(
            "recall", recall, sample_size, round=4
        )
        fairseq.metrics.log_scalar( 
            "f1", f1, sample_size, round=4
        )
        if ntokens != 0:
            rr = npredpos / ntokens
        else:
            rr = 0
        fairseq.metrics.log_scalar(
            "retrieve_ratio" , rr,  sample_size, round=4
        )
        # if sample_size != ntokens:
        #     fairseq.metrics.log_scalar(
        #         "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
        #     )
        #     fairseq.metrics.log_derived(
        #         "ppl", lambda meters: fairseq.utils.get_perplexity(meters["nll_loss"].avg)
        #     )
        # else:
        #     fairseq.metrics.log_derived(
        #         "ppl", lambda meters: fairseq.utils.get_perplexity(meters["loss"].avg)
        #     )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
MODEL_NAME = "less_retrieve_knn_mt"

@register_model(MODEL_NAME)
class LessRetrieveKNNMT(TransformerModel):
    r"""
    The vanilla knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "test_metrics", "inference", "train_less_retrieve"],
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
        parser.add_argument("--whether_retrieve_selector_path", help="Path to save/load the selector checkpoint", required=True)
        parser.add_argument("--use_mt_loss_for_selector", help="Whether to use matchine translation loss to optimize selector", 
                            default=True, type=lambda x : x in ['y', "Y", "Yes", "T", "True"])
        parser.add_argument("--gumbel_max_tau", help="When use_mt_loss_for_selector is on, set max tau hyper-parameter for gumbel softmax", default=0.1)
        parser.add_argument("--gumbel_min_tau", help="When use_mt_loss_for_selector is on, set min tau hyper-parameter for gumbel softmax", default=0.1)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with VanillaKNNMTDecoder
        """
        return LessRetrieveKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
        
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.decoder.parameters(recurse=recurse)
    
    def after_inference_hook(self):
        if hasattr(self.decoder, "after_inference_hook"):
            self.decoder.after_inference_hook()
            
    def after_train_hook(self):
        if hasattr(self.decoder, "after_train_hook"):
            self.decoder.after_train_hook()

class LessRetrieveKNNMTDecoder(TransformerDecoder):
    r"""
    The vanilla knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        selector_impl = LessRetrieveSelectorSimple
        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]
        elif args.knn_mode == 'test_metrics':
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=["vals"])
            self.datastore.load_faiss_index("keys")
            self.combiner = Combiner(lambda_=args.knn_lambda,
                        temperature=args.knn_temperature, probability_dim=len(dictionary))
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.whether_retrieve_selector = selector_impl.load(args.whether_retrieve_selector_path)
            self.token_meet_cnt = nn.Parameter(torch.zeros(embed_tokens.weight.shape[0], dtype=torch.float), requires_grad=False)
            self.tp = 0
            self.fp = 0
            self.tn = 0
            self.fn = 0
            self.num_predict_retrieve = 0
            self.num_total_tokens = 0
        else:
            if args.knn_mode in ["train_less_retrieve", "inference"]:
                self.datastore = Datastore.load(args.knn_datastore_path, load_list=["vals"])
                self.datastore.load_faiss_index("keys")
                self.combiner = Combiner(lambda_=args.knn_lambda,
                        temperature=args.knn_temperature, probability_dim=len(dictionary))
                self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
                self.gumbel = GumbelSoftmaxLayer(args.max_epoch, init_tau=args.gumbel_max_tau, min_tau=args.gumbel_min_tau)
            
                if args.knn_mode == "inference":
                    self.whether_retrieve_selector = selector_impl.load(args.whether_retrieve_selector_path)
                    
                else:
                    #self.whether_retrieve_selector = LessRetrieveSelector(self.output_projection.in_features + embed_tokens.weight.shape[0], 2048)
                    self.whether_retrieve_selector = selector_impl(self.output_projection.in_features, self.output_projection.in_features)
                    if not args.use_mt_loss_for_selector:
                        logging.info("Do not use translation loss")  
        self.retrieve_timer = StopwatchMeter()   
                   
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
        target : torch.Tensor = None
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
        extra["args"] = self.args

        if self.args.knn_mode == "build_datastore":
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            # save half precision keys
            self.datastore["keys"].add(keys.half())
            
        if self.args.knn_mode == 'train_less_retrieve':
            # Retrive all the time when training less retrieve
            self.retriever.retrieve(z, return_list=["vals", "distances"])
             
        if not features_only:
            x = self.output_layer(z)
        
        if self.args.knn_mode == 'train_less_retrieve':
           
            selector_label = (torch.argmax(torch.softmax(x, dim=-1), dim=-1) == target).long()

            selector_pred_prob = self.whether_retrieve_selector(z)

            if self.args.use_mt_loss_for_selector:
                self.selector_gumbel_prob = self.gumbel(selector_pred_prob)
                self.selector_pred = (torch.argmax(self.selector_gumbel_prob, dim=-1) == 0)
            else:
                self.selector_gumbel_prob = None
                self.selector_pred = (torch.softmax(selector_pred_prob, dim=-1).argmax(dim=-1) == 0)
            
            B, T, H = z.shape
            
            selected_features = z.reshape(B*T,H)[self.selector_pred.view(B*T)].view(-1,H).unsqueeze(1)
            
            self.retriever.retrieve(selected_features, return_list=["vals", "distances"])
            
            return x, extra, {"selector_label" : selector_label, "selector_pred" : selector_pred_prob}
        elif self.args.knn_mode =='inference':
            # Decide whether to retrieve
            #record_timer_start(self.retrieve_timer)
            
            # Let the selector predict for each token
            selector_pred_prob = self.whether_retrieve_selector(z)   
            self.selector_pred = (torch.softmax(selector_pred_prob, dim=-1).argmax(dim=-1) == 0)  # Eq. 8
       
            # select probabilities that requires retrievals
            B, T, H = z.shape
            selected_features = z.reshape(B*T,H)[self.selector_pred.view(B*T)].view(-1,H).unsqueeze(1)  
            
            # Retrievel only for partial tokens.
            self.retriever.retrieve(selected_features, return_list=["vals", "distances"])
            #record_timer_end(self.retrieve_timer)
            return x, extra, {}
        elif self.args.knn_mode == 'test_metrics':
            #self.retriever.retrieve(z, return_list=["vals", "distances"])
           # knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=x.device)
           # combined_prob, _ = self.combiner.get_combined_prob(knn_prob, x, log_probs=True)  #log_probs is True
            
            # net_pred = torch.argmax(torch.softmax(x, dim=-1),dim=-1)
            # net_correct = (net_pred == target).int()
            
            selector_pred_prob = self.whether_retrieve_selector(z)
            
            selector_pred = (torch.softmax(selector_pred_prob, dim=-1)).argmax(dim=-1)
            selector_label = (torch.argmax(torch.softmax(x, dim=-1), dim=-1) == target).long()

            pos_flag = (selector_label.view(-1) == 0)
            neg_flag = (selector_label.view(-1) == 1)
            pred_pos = (selector_pred.view(-1) == 0)
            pred_neg = (selector_pred.view(-1) == 1)
            tp = torch.logical_and(pred_pos, pos_flag).int().sum().item()
            fp = torch.logical_and(pred_pos, neg_flag).int().sum().item()
            tn = torch.logical_and(pred_neg, neg_flag).int().sum().item()
            fn = torch.logical_and(pred_neg, pos_flag).int().sum().item()
            self.tp += tp 
            self.fp += fp 
            self.tn += tn 
            self.fn += fn
            
            self.selector_pred = (selector_pred == 0)
            
            self.num_predict_retrieve += (self.selector_pred).int().sum().item()
            
            B, T, H = z.shape
            
            self.num_total_tokens += B * T
            
            selected_features = z.reshape(B*T,H)[self.selector_pred.view(B*T)].view(-1,H).unsqueeze(1)
            
            self.retriever.retrieve(selected_features, return_list=["vals", "distances"])
            return x, extra, {}
                

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
        if self.args.knn_mode == "inference" or self.args.knn_mode == 'train_less_retrieve' or self.args.knn_mode == "test_metrics":
            if self.retriever.results['distances'].shape[0] > 0:
                
                # Combine probability as Eq. 2 and Eq. 5
                #self.retrieve_timer.start()
                B, T, V = net_output[0].shape
                knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)  # Eq. 1
                    
                if self.args.knn_mode == 'train_less_retrieve' and self.args.use_mt_loss_for_selector:
                    selected_net_prob = torch.mul(self.selector_gumbel_prob[self.selector_pred][:,0].unsqueeze_(1), net_output[0][self.selector_pred]).unsqueeze_(1)
                else:  # Gumbel softmax is disable when infernece or mt_loss_for_selector is turned off.
                    selected_net_prob = net_output[0][self.selector_pred].unsqueeze_(1)
                    
                combined_prob, _ = self.combiner.get_combined_prob(knn_prob, selected_net_prob, log_probs=True)  # Eq. 2
                final_prob = torch.clone(net_output[0]).reshape(B*T,-1) 

                # Eq. 5
                final_prob[self.selector_pred.view(B*T)] = combined_prob.view(-1, V)
                final_prob = final_prob.view(B, T, -1)
                
                # Keep the probability in log-space
                if log_probs:
                    res = torch.log_softmax(final_prob, dim=-1)
                else:
                    res = torch.softmax(final_prob, dim=-1)
                #self.retrieve_timer.stop()
                return res
            else:
                #self.retrieve_timer.start()
                knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
                #combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
                if knn_prob.shape[0] > 0:
                    combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
                    self.retrieve_timer.stop()
                    return combined_prob
                else:
                    if log_probs:
                        res = F.log_softmax(net_output[0], dim=-1)
                    else:
                        res = torch.softmax(net_output[0], dim=-1)
                    #self.retrieve_timer.stop()
                    return res
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)
        
        
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.whether_retrieve_selector.parameters(recurse=True)
    
    def after_inference_hook(self):
        if self.retrieve_timer.start_time is None:
            print("KNN overhead time is not recoreded")
        else:
            print(f"KNN overhead time = {self.retrieve_timer.sum}s")
    
    def after_train_hook(self):
        if self.args.knn_mode == 'test_metrics':
            tp = self.tp 
            fp = self.fp 
            tn = self.tn 
            fn = self.fn 
            if (tp + fp + tn + fn) != 0:
                accuracy = (tp + tn) / (tp + fp + tn + fn)
            else:
                accuracy = 0
                
            if tp + fp != 0:
                precision = tp / (tp + fp)
            else:
                precision = 0

            if tp + fn != 0:
                recall = tp / (tp + fn)
            else:
                recall = 0 
            
            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
                
            retrieving_ratio = self.num_predict_retrieve / self.num_total_tokens
            print(f"Test Metrics -- Acc : {accuracy}, P : {precision}, R : {recall}, F1 : {f1}, Retrieving Ratio = {retrieving_ratio}")


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
    

        

