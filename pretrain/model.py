import torch.nn as nn
from einops import rearrange
import torch
import numpy as np
import math

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor,
)
from modules.sequence_modeling import BidirectionalLSTM


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
        self.SelfSL_layer = SelfSL_layer

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
        else:
            raise Exception("No FeatureExtraction module specified")
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
                    )
                )
                self.SequenceModeling_output = opt.hidden_size
            else:
                print("No SequenceModeling module specified")
                self.SequenceModeling_output = self.FeatureExtraction_output

        if not SelfSL_layer:
            raise

        else:
            """for self-supervised learning (SelfSL)"""
            if self.opt.self == "RotNet" or self.opt.self == "MoCo":
                self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 1))  # make width -> 1
            elif self.opt.self == "MoCoSeqCLR":
                if self.opt.instance_map == "window":
                    self.AdaptiveAvgPool_subword = nn.AdaptiveAvgPool2d((None, 4))  # make width -> 4 subwords
                elif self.opt.instance_map == "all":
                    self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 1))  # make width -> 1 word
            else:
                raise NotImplementedError
            if SelfSL_layer == "CNN":
                self.SelfSL_FFN_input = self.FeatureExtraction_output
            if SelfSL_layer == "CNNLSTM":
                self.SelfSL_FFN_input = self.SequenceModeling_output

            if "MoCo" in self.opt.self:
                self.frame_fc = nn.Linear(
                    self.SelfSL_FFN_input, opt.moco_dim
                )  # 128 is used for MoCo paper.
                self.subword_fc = nn.Linear(
                    self.SelfSL_FFN_input, opt.moco_dim
                )  # 128 is used for MoCo paper.
                self.word_fc = nn.Linear(
                    self.SelfSL_FFN_input, opt.moco_dim
                )  # 128 is used for MoCo paper.              

            self.AdaptiveAvgPool_word = nn.AdaptiveAvgPool2d((None, 1))

            self.frame_count = 26
            self.subword_count = 4
            self.word_count = 1

            if self.opt.permutation:
                self.permutation_block_count = 2
                self.permutation_img_count = self.opt.permutation_img_count
    

    def undo_permutation(self, permutated_view, permutation_index):
        """
        Args:
            permutated_view: torch.Tensor([N, T, D])
            permutated_index: np.array([permutation_count])
        Returns:
            restored_view: torch.Tensor(N, T, D)
        """
        # id permutation
        permutation_count = self.permutation_block_count * self.permutation_img_count

        N = permutated_view.size(0)
        T = permutated_view.size(1)
        D = permutated_view.size(2)
        permutated_view = permutated_view.reshape(N // self.permutation_img_count, self.permutation_img_count, T, D)
        permutated_view = permutated_view.reshape(N // self.permutation_img_count, self.permutation_img_count * T, D)
        permutated_view = permutated_view.reshape(N // self.permutation_img_count, permutation_count, T // self.permutation_block_count, D)

        restore_index = np.argsort(permutation_index)
        restored_view = permutated_view[:, restore_index, :, :]
        
        restored_view = restored_view.reshape(N // self.permutation_img_count, self.permutation_img_count * T, D)
        restored_view = restored_view.reshape(N // self.permutation_img_count, self.permutation_img_count, T, D)
        restored_view = restored_view.reshape(N, T, D)
        
        return restored_view
            

    def forward(self, image, frame_select_index, permutation_index):
        SelfSL_layer = self.SelfSL_layer
        """Transformation stage"""
        if not self.stages["Trans"] == "None":
            with torch.no_grad():
                image = self.Transformation(image)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(image)
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
            prediction_SelfSL = self.fc(
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

            if self.opt.permutation:
                if permutation_index is None:
                    pass
                else:
                    contextual_feature = self.undo_permutation(contextual_feature, permutation_index)

            # frame
            frame_feature = contextual_feature.permute(0, 2, 1)  # [b, w, c]  -> [b, c, w]  
            
            # subword
            subword_feature = self.AdaptiveAvgPool_subword(frame_feature)  # [b, c, t]           

            # word
            word_feature = self.AdaptiveAvgPool_word(frame_feature)  # [b, c, w] -> [b, c, 1]
            
            frame_feature = rearrange(frame_feature, 'b c t -> b t c') 
            frame_ssl = self.frame_fc(frame_feature)
            subword_feature = rearrange(subword_feature, 'b c t -> b t c')
            subword_ssl = self.subword_fc(subword_feature)
            word_feature = rearrange(word_feature, 'b c t -> b t c')
            word_ssl = self.word_fc(word_feature)
            
            return (frame_ssl, subword_ssl, word_ssl)
