import torch
from move_classifier import rec_common_ground, hyperparam, modalities

class moveFeature():
    def __init__(self):
        self.model = torch.load("move_gnn_01.pt")
    def processFrame(self, in_bert, in_open, in_cps, in_action, in_gamr):
        N = 4
        in_bert = torch.zeros((N, 768)) # N is the length of the sequence (the utterance of interest + 3 previous utterances for context)
        in_open = torch.zeros((N, 88))
        in_cps = torch.zeros((N, 3))
        in_action = torch.zeros((N, 78))
        in_gamr = torch.zeros((N, 896))
        return self.model(in_bert, in_open, in_cps, in_action, in_gamr, hyperparam, modalities)
        