import re
from itertools import product

class CommonGround():

    def __init__(self):
        # initialize possible weights of blocks
        self.poss = {'red': {10, 20, 30, 40, 50},
                     'blue': {10, 20, 30, 40, 50},
                     'green': {10, 20, 30, 40, 50},
                     'purple': {10, 20, 30, 40, 50},
                     'yellow': {10, 20, 30, 40, 50}}
        # initialize evidence for weights (empty)
        self.evidence_for = {'red': set(),
                             'blue': set(),
                             'green': set(),
                             'purple': set(),
                             'yellow': set()}
        # initialize evidence against weights (empty)
        self.evidence_against = {'red': set(),
                                 'blue': set(),
                                 'green': set(),
                                 'purple': set(),
                                 'yellow': set()}
        # initialize QBank (Cartesian product of blocks and weights),
        # EBank (empty), and FBank (empty)
        self.generate_banks()

    def print(self):
        print('self.poss:', self.poss)
        print('self.evidence_for:', self.evidence_for)
        print('self.evidence_against:', self.evidence_against)
        print('self.qbank:', self.qbank)
        print('self.ebank:', self.ebank)
        print('self.fbank:', self.fbank)

    def generate_banks(self):
        # generate banks based on self.poss, self.evidence_for, and
        # self.evidence_against
        self.qbank = set()
        self.ebank = set()
        self.fbank = set()
        for block in ('red', 'blue', 'green', 'purple', 'yellow'):
            # if only one possible weight, add block=weight to FBank
            if len(self.poss[block]) == 1:
                (weight,) = self.poss[block]
                self.fbank.add(block + '=' + str(weight))
            else:
                # add evidence to EBank
                for weight in self.evidence_for[block]:
                    self.ebank.add(block + '=' + str(weight))
                for weight in self.evidence_against[block]:
                    self.ebank.add(block + '!=' + str(weight))
                # add non-evidence possibilities to QBank
                for weight in (self.poss[block]
                               .difference(self.evidence_for[block])
                               .difference(self.evidence_against[block])):
                    self.qbank.add(block + '=' + str(weight))

    def update(self, move, content):
        # content is one or more props connected by ' and '
        props = re.split(r'\s+and\s+', content)
        for prop in props:
            # prop consists of block, relation, and rhs
            prop_match = re.match(r'(red|blue|green|yellow|purple)\s*(=|<|>|!=)\s*(.*)', prop)
            if prop_match:
                block = prop_match[1]
                relation = prop_match[2]
                rhs = prop_match[3]
                # rhs contains weight or list of blocks (assumed connected by +)
                rhs_weight_match = re.search(r'10|20|30|40|50', rhs)
                if rhs_weight_match:
                    rhs_weight = int(rhs_weight_match[0])
                else:
                    rhs_weight = 0
                rhs_blocks = re.findall(r'red|blue|green|yellow|purple', rhs)
                # if rhs contains a single block and we know more about block
                # than rhs_block, switch them
                if (len(rhs_blocks) == 1
                    and (len(self.poss[block]) < len(self.poss[rhs_blocks[0]])
                    or (len(self.poss[block]) == len(self.poss[rhs_blocks[0]])
                    and len(self.evidence_for[block].union(
                        self.evidence_against[block])) >
                        len(self.evidence_for[rhs_blocks[0]].union(
                            self.evidence_against[rhs_blocks[0]]))))):
                    block, rhs_blocks = rhs_blocks[0], [block]
                # STATEMENTs add weights to evidence (for or against)
                if move == 'STATEMENT':
                    if relation == '=':
                        # if block = weight
                        if rhs_weight:
                            # add weight to self.evidence_for[block]
                            self.evidence_for[block].add(rhs_weight)
                        # elif block = sum(blocks)
                        elif rhs_blocks:
                            # if rhs contains a single block, update
                            # self.evidence_for[block] with
                            # self.evidence_for[rhs_block] and
                            # self.evidence_against[block] with
                            # self.evidence_against[rhs_block]
                            if len(rhs_blocks) == 1:
                                for poss_weight in (
                                    self.evidence_for[rhs_blocks[0]]):
                                    self.evidence_for[block].add(poss_weight)
                                for not_weight in (
                                    self.evidence_against[rhs_blocks[0]]):
                                    self.evidence_against[block].add(not_weight)
                            else:
                                # find possible values of sum(blocks)
                                poss_blocks = list(self.poss[rhs_block]
                                                   for rhs_block in rhs_blocks)
                                poss_weights = set(sum(weights) for weights
                                                   in product(*poss_blocks))
                                # add impossible weights to
                                # self.evidence_against[block]
                                for not_weight in ({10, 20, 30, 40, 50}
                                                   .difference(poss_weights)):
                                    self.evidence_against[block].add(not_weight)
                                # find evidenced values of sum(blocks)
                                ev_blocks = list(self.evidence_for[rhs_block]
                                                 for rhs_block in rhs_blocks)
                                ev_weights = set(sum(weights) for weights
                                                 in product(*ev_blocks))
                                # add evidenced weights to
                                # self.evidence_for[block]
                                for poss_weight in (self.poss[block]
                                                    .intersection(ev_weights)):
                                    self.evidence_for[block].add(poss_weight)
                    elif relation == '<':
                        # if block < weight
                        if rhs_weight:
                            # find possible weights
                            poss_weights = filter(lambda x: x < rhs_weight,
                                                  self.poss[block])
                            # add impossible weights to
                            # self.evidence_against[block]
                            for not_weight in {10, 20, 30, 40, 50}.difference(
                                poss_weights):
                                self.evidence_against[block].add(not_weight)
                        # elif block < sum(blocks)
                        elif rhs_blocks:
                            # find possible values of sum(blocks)
                            poss_blocks = list(self.poss[rhs_block]
                                               for rhs_block in rhs_blocks)
                            poss_weights = set(sum(weights)
                                        for weights in product(*poss_blocks))
                            # weight < max(poss_weights)
                            if len(poss_weights) > 1:
                                poss_weights_2 = filter(
                                    lambda x: x < max(poss_weights),
                                    self.poss[block])
                            else:
                                poss_weights_2 = poss_weights
                            # add impossible weights to
                            # self.evidence_against[block]
                            for not_weight in {10, 20, 30, 40, 50}.difference(
                                poss_weights_2):
                                self.evidence_against[block].add(not_weight)
                    elif relation == '>':
                        # if block > weight
                        if rhs_weight:
                            # find possible weights
                            poss_weights = filter(lambda x: x > rhs_weight,
                                                  self.poss[block])
                            # add impossible weights to
                            # self.evidence_against[block]
                            for not_weight in {10, 20, 30, 40, 50}.difference(
                                poss_weights):
                                self.evidence_against[block].add(not_weight)
                        # elif block > sum(blocks)
                        elif rhs_blocks:
                            # find possible values of sum(blocks)
                            poss_blocks = list(self.poss[rhs_block]
                                               for rhs_block in rhs_blocks)
                            poss_weights = set(sum(weights)
                                        for weights in product(*poss_blocks))
                            # weight > min(poss_weights)
                            if len(poss_weights) > 1:
                                poss_weights_2 = filter(
                                    lambda x: x > min(poss_weights),
                                    self.poss[block])
                            else:
                                poss_weights_2 = poss_weights
                            # add impossible weights to
                            # self.evidence_against[block]
                            for not_weight in {10, 20, 30, 40, 50}.difference(
                                poss_weights_2):
                                self.evidence_against[block].add(not_weight)
                    elif relation == '!=':
                        # if block != weight
                        if rhs_weight:
                            # add weight to self.evidence_against[block]
                            self.evidence_against[block].add(rhs_weight)
                        # elif block != sum(blocks)
                        elif rhs_blocks:
                            # find impossible values of sum(blocks)
                            poss_blocks = list(self.poss[rhs_block]
                                               for rhs_block in rhs_blocks)
                            poss_weights = set(sum(weights)
                                        for weights in product(*poss_blocks))
                            # if only one impossible weight, add it to
                            # self.evidence_against[block]
                            if len(poss_weights) == 1:
                                (not_weight,) = poss_weights
                                self.evidence_against[block].add(not_weight)
                # ACCEPTs remove impossible weights
                elif move == 'ACCEPT':
                    if relation == '=':
                        # if block = weight
                        if rhs_weight:
                            # remove impossible weights
                            for not_weight in ({10, 20, 30, 40, 50}
                                               .difference({rhs_weight})):
                                self.poss[block].discard(not_weight)
                                self.evidence_for[block].discard(not_weight)
                                self.evidence_against[block].discard(not_weight)
                            # update possible weights
                            self.poss[block].add(rhs_weight)
                            self.evidence_for[block].add(rhs_weight)
                            self.evidence_against[block].discard(rhs_weight)
                        # elif block = sum(blocks)
                        elif rhs_blocks:
                            # if rhs contains a single block, update
                            # self.poss[block] with self.poss[rhs_block],
                            # self.evidence_for[block] with
                            # self.evidence_for[rhs_block]
                            # self.evidence_against[block] with
                            # self.evidence_against[rhs_block]
                            if len(rhs_blocks) == 1:
                                self.poss[block] = self.poss[rhs_blocks[0]]
                                self.evidence_for[block] = (
                                    self.evidence_for[rhs_blocks[0]])
                                self.evidence_against[block] = (
                                    self.evidence_against[rhs_blocks[0]])
                            else:
                                # find possible values of sum(blocks)
                                poss_blocks = list(self.poss[rhs_block]
                                                   for rhs_block in rhs_blocks)
                                poss_weights = set(sum(weights)
                                        for weights in product(*poss_blocks))
                                # remove impossible weights
                                for not_weight in ({10, 20, 30, 40, 50}
                                                   .difference(poss_weights)):
                                    self.poss[block].discard(not_weight)
                                    self.evidence_for[block].discard(not_weight)
                                    self.evidence_against[block].discard(
                                        not_weight)
                                # update possible weights
                                poss_weights_2 = poss_weights.intersection(
                                    {10, 20, 30, 40, 50})
                                for poss_weight in poss_weights_2:
                                    self.poss[block].add(poss_weight)
                                    # if only one possible weight, add it to
                                    # self.evidence_for[block] and remove it
                                    # from self.evidence_against[block]
                                    if len(poss_weights_2) == 1:
                                        self.evidence_for[block].add(
                                            poss_weight)
                                        self.evidence_against[block].discard(
                                            poss_weight)
                    elif relation == '<':
                        # if block < weight
                        if rhs_weight:
                            # find possible weights
                            poss_weights = filter(lambda x: x < int(rhs_weight),
                                                  self.poss[block])
                            # remove impossible weights
                            for not_weight in ({10, 20, 30, 40, 50}
                                               .difference(poss_weights)):
                                self.poss[block].discard(not_weight)
                                self.evidence_for[block].discard(not_weight)
                                self.evidence_against[block].discard(not_weight)
                        # elif block < sum(blocks)
                        elif rhs_blocks:
                            # find possible values of sum(blocks)
                            poss_blocks = list(self.poss[rhs_block]
                                               for rhs_block in rhs_blocks)
                            poss_weights = set(sum(weights)
                                        for weights in product(*poss_blocks))
                            # weight < max(poss_weights)
                            if len(poss_weights) > 1:
                                poss_weights_2 = filter(
                                    lambda x: x < max(poss_weights),
                                    self.poss[block])
                            else:
                                poss_weights_2 = poss_weights
                            # remove impossible weights
                            for not_weight in ({10, 20, 30, 40, 50}
                                               .difference(poss_weights_2)):
                                self.poss[block].discard(not_weight)
                                self.evidence_for[block].discard(not_weight)
                                self.evidence_against[block].discard(not_weight)
                    elif relation == '>':
                        # if block > weight
                        if rhs_weight:
                            # find possible weights
                            poss_weights = filter(lambda x: x > int(rhs_weight),
                                                  self.poss[block])
                            # remove impossible weights
                            for not_weight in ({10, 20, 30, 40, 50}
                                               .difference(poss_weights)):
                                self.poss[block].discard(not_weight)
                                self.evidence_for[block].discard(not_weight)
                                self.evidence_against[block].discard(not_weight)
                        # elif block > sum(blocks)
                        elif rhs_blocks:
                            # find possible values of sum(blocks)
                            poss_blocks = list(self.poss[rhs_block]
                                               for rhs_block in rhs_blocks)
                            poss_weights = set(sum(weights)
                                        for weights in product(*poss_blocks))
                            # weight > min(poss_weights)
                            if len(poss_weights) > 1:
                                poss_weights_2 = filter(
                                    lambda x: x > min(poss_weights),
                                    self.poss[block])
                            else:
                                poss_weights_2 = poss_weights
                            # remove impossible weights
                            for not_weight in ({10, 20, 30, 40, 50}
                                               .difference(poss_weights_2)):
                                self.poss[block].discard(not_weight)
                                self.evidence_for[block].discard(not_weight)
                                self.evidence_against[block].discard(not_weight)
                    elif relation == '!=':
                        # if block != weight
                        if rhs_weight:
                            # remove impossible weight
                            self.poss[block].discard(rhs_weight)
                            self.evidence_for[block].discard(rhs_weight)
                            self.evidence_against[block].discard(rhs_weight)
                        # elif block != sum(blocks)
                        elif rhs_blocks:
                            # find impossible values of sum(blocks)
                            poss_blocks = (self.poss[rhs_block]
                                               for rhs_block in rhs_blocks)
                            poss_weights = (set(sum(weights)
                                        for weights in product(*poss_blocks)))
                            # if only one impossible weight, remove it
                            if len(poss_weights) == 1:
                                (not_weight,) = poss_weights
                                self.poss[block].discard(not_weight)
                                self.evidence_for[block].discard(not_weight)
                                self.evidence_against[block].discard(not_weight)
                # else, pass
                # OBSERVATIONs, INFERENCEs, and RECOMMENDATIONs do nothing
                # things to watch out for:
                # - QUESTIONs and ANSWERs
                # - DOUBTs after ACCEPTs
                else:
                    pass
        # update banks
        self.generate_banks()
