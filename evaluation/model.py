from doctest import debug_script
import torch.nn as nn
from einops import rearrange

from modules.backbone import ResTransformer
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor,
)
from modules.vit_feature_extraction import ViT_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.scatter_decoder import ScatterDecoder
from modules.attention import *

import torchvision.transforms as transforms
import random
import torch


class Model(nn.Module):
    def __init__(self, opt, SelfSL_layer=False):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {
            "Trans": opt.Transformation,
            "Feat": opt.FeatureExtraction,
            "Seq": opt.SequenceModeling,
            "Pred": opt.Prediction,
        }

        """ Transformation """
        if opt.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial,
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel,
            )
        else:
            print("No Transformation module specified")

        """ FeatureExtraction """
        if opt.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "ResTransformer":
            self.FeatureExtraction = ResTransformer()
        elif opt.FeatureExtraction == "ViT":
            self.FeatureExtraction = ViT_FeatureExtractor(
                image_size=(32,100),
                patch_size=4,
                dim=192,
                depth=6,
                heads=8,
                mlp_dim=1024,
                dropout=0.1,
                dim_head=32,
                emb_dropout=0.1
            )
        else:
            raise Exception("No FeatureExtraction module specified")
        if opt.FeatureExtraction == "ViT":
            self.FeatureExtraction_output = 192
        else:
            self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        if not SelfSL_layer or SelfSL_layer == "CNNLSTM":  # for STR or CNNLSTM SSL
            """Sequence modeling"""
            if opt.SequenceModeling == "BiLSTM":
                self.SequenceModeling = nn.Sequential(
                    BidirectionalLSTM(
                        self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size
                    ),
                    BidirectionalLSTM(
                        opt.hidden_size, opt.hidden_size, opt.hidden_size
                    ),
                )
                self.SequenceModeling_output = opt.hidden_size
            else:
                print("No SequenceModeling module specified")
                self.SequenceModeling_output = self.FeatureExtraction_output

        if not SelfSL_layer:  # for STR.
            """Prediction"""
            if opt.Prediction == "CTC":
                self.Prediction = nn.Sequential(
                    BidirectionalLSTM(
                        self.SequenceModeling_output, opt.hidden_size, opt.hidden_size
                    ),
                    BidirectionalLSTM(
                        opt.hidden_size, opt.hidden_size, opt.hidden_size
                    ),
                    nn.Linear(opt.hidden_size, opt.num_class)
                )
            elif opt.Prediction == "Attn":
                # self.Prediction = Attention(
                #     self.SequenceModeling_output, opt.hidden_size, opt.num_class
                # )
                self.ScatterDecoder = ScatterDecoder(self.FeatureExtraction_output, opt.hidden_size, opt.num_class)
            elif opt.Prediction == "Attention":
                self.attention = PositionAttention(
                    max_length=self.opt.batch_max_length + 1,  # additional stop token
                    mode='nearest',
                    )
                self.cls = nn.Linear(self.FeatureExtraction_output, opt.num_class)

            else:
                raise Exception("Prediction is neither CTC or Attn")
            

        else:
            """for self-supervised learning (SelfSL)"""
            if self.opt.self == "RotNet" or self.opt.self == "MoCo":
                self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 1))  # make width -> 1
            elif self.opt.self == "MoCoSeqCLR":
                self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 5))  # make width -> 5 instances
            else:
                raise NotImplementedError
            if SelfSL_layer == "CNN":
                self.SelfSL_FFN_input = self.FeatureExtraction_output
            if SelfSL_layer == "CNNLSTM":
                self.SelfSL_FFN_input = self.SequenceModeling_output

            if "RotNet" in self.opt.self:
                self.SelfSL = nn.Linear(
                    self.SelfSL_FFN_input, 4
                )  # 4 = [0, 90, 180, 270] degrees
            elif "MoCo" in self.opt.self:
                self.SelfSL = nn.Linear(
                    self.SelfSL_FFN_input, 128
                )  # 128 is used for MoCo paper.
            

    def forward(self, image, text=None, is_train=True, SelfSL_layer=False):
        """Transformation stage"""
        if not self.stages["Trans"] == "None":
            original_image_batch = image.cpu().clone()
            image = self.Transformation(image)
            # for debug
            # debug_image_batch = image.cpu().clone()
            # debug_image_batch = torch.cat((original_image_batch, debug_image_batch), -1)
            # debug_image_batch.mul_(0.5).add_(0.5)  # -> (0,1)
            # for i in range(len(debug_image_batch)):
            #     debug_image = debug_image_batch[i]
            #     debug_image = transforms.ToPILImage()(debug_image)
            #     debug_image.save("debug/debug{}.jpg".format(i))
            # raise NotImplementedError


        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(image)
        if self.stages["Pred"] == "Attention":
            visual_feature = nn.functional.pad(visual_feature, (3, 4, 0, 0), "replicate")
            attn_vecs, attn_scores = self.attention(visual_feature)  # (N, T, E), (N, T, H, W)
            logits = self.cls(attn_vecs) # (N, T, C)
            return logits
        if self.opt.FeatureExtraction == "ViT":
            assert visual_feature.size(-1) == 25 and visual_feature.size(-2) == 8
            visual_feature = nn.functional.pad(visual_feature, (0, 1, 0, 0), "replicate")
        visual_feature = visual_feature.permute(
            0, 3, 1, 2
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(
            visual_feature
        )  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)  # [b, w, c, 1] -> [b, w, c]

        """ for self supervised learning on Feature extractor (CNN part) """
        if SelfSL_layer == "CNN":
            visual_feature = visual_feature.permute(0, 2, 1)  # [b, w, c] -> [b, c, w]
            visual_feature = self.AdaptiveAvgPool_2(
                visual_feature
            )  # [b, c, w] -> [b, c, t]
            visual_feature = rearrange(visual_feature, 'b c t -> (b t) c')
            # visual_feature = visual_feature.squeeze(2)  # [b, c, 1] -> [b, c]
            prediction_SelfSL = self.SelfSL(
                visual_feature
            )  # [b, c] -> [b, SelfSL_class]
            return prediction_SelfSL

        """ Sequence modeling stage """
        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(
                visual_feature
            )  # [b, num_steps, opt.hidden_size]
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        if SelfSL_layer == "CNNLSTM":
            contextual_feature = contextual_feature.permute(0, 2, 1)  # [b, w, c] -> [b, c, w]
            contextual_feature = self.AdaptiveAvgPool_2(
                contextual_feature
            )  # [b, c, w] -> [b, c, t]
            contextual_feature = rearrange(contextual_feature, 'b c t -> (b t) c')
            # contextual_feature = contextual_feature.squeeze(2)  # [b, c, 1] -> [b, c]
            prediction_SelfSL = self.SelfSL(
                contextual_feature
            )  # [b, c] -> [b, SelfSL_class]
            return prediction_SelfSL
        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            # prediction = self.Prediction(
            #     contextual_feature.contiguous(),
            #     text,
            #     is_train,
            #     batch_max_length=self.opt.batch_max_length,
            # )
            prediction = self.ScatterDecoder(visual_feature, contextual_feature, text, is_train, self.opt.batch_max_length)

        return prediction  # [b, num_steps, opt.num_class]
