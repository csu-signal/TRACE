import os
from unified_planning.io import PDDLWriter
from unified_planning.shortcuts import *
# from unified_planning.engines import PlanGenerationResultStatus
import subprocess
up.shortcuts.get_environment().credits_stream = None


def weight_name(weight):
    # print(int(weight.name[:-1]))
    return int(weight.name[:-1])

def create_problem():

    problem = Problem('weights')

    Block = UserType('block')
    Weight = UserType('weight')
    Participant = UserType('participant')

    participants = [Object('participant1', Participant), Object('participant2', Participant), Object('participant3', Participant)]

    # CHANGED: added weights as objects to avoid numerical planning
    weights_list = [10, 20, 30, 40, 50]
    weights = [Object(str(weight) + 'g', Weight) for weight in weights_list]

    # assert that the numbers in weights_list are multiples of ten and increase by ten
    for i in range(len(weights_list) - 1):
        assert weights_list[i] % 10 == 0
        assert weights_list[i + 1] - weights_list[i] == 10

    actual_weight = Fluent('actual_weight', block=Block, weight=Weight)
    believed_weight = Fluent('believed_weight', block=Block, weight=Weight, participant=Participant)

    heavier = Fluent('heavier', heavier=Block, lighter=Block, participant=Participant)

    pay_attention = InstantaneousAction('pay_attention', participant=Participant)
    is_paying_attention = Fluent('is_paying_attention', participant=Participant)

    participant = pay_attention.parameter('participant')
    pay_attention.add_effect(is_paying_attention(participant), True)

    stop_paying_attention = InstantaneousAction('stop_paying_attention', participant=Participant)
    participant = stop_paying_attention.parameter('participant')
    stop_paying_attention.add_effect(is_paying_attention(participant), False)

    set_weight = InstantaneousAction('set_weight', block=Block, weight=Weight)
    block = set_weight.parameter('block')
    weight = set_weight.parameter('weight')

    set_weight.add_effect(actual_weight(block, weight), True)

    # precondition, no weight is already set
    for w in weights:
        set_weight.add_precondition(Not(actual_weight(block, w)))


    compare = InstantaneousAction('compare', left=Block, right=Block)
    left = compare.parameter('left')
    right = compare.parameter('right')

    compare.add_precondition(Not(Equals(left, right)))

    # one to one, if both sides are equal

    # CHANGED: added weights as bool
    for participant in participants:
        for w in weights:
            compare.add_effect(
                believed_weight(left, w, participant),
                True,
                condition=And(
                    believed_weight(right, w, participant),
                    actual_weight(left, w),
                    actual_weight(right, w),
                    is_paying_attention(participant)
                )
            )

            compare.add_effect(
                believed_weight(right, w, participant),
                True,
                condition=And(
                    believed_weight(left, w, participant),
                    actual_weight(left, w),
                    actual_weight(right, w),
                    is_paying_attention(participant)
                )
            )
    compare_one_two = InstantaneousAction('compare_one_two', left=Block, right1=Block, right2=Block)
    left = compare_one_two.parameter('left')
    right1 = compare_one_two.parameter('right1')
    right2 = compare_one_two.parameter('right2')

    compare_one_two.add_precondition(Not(Equals(left, right1)))
    compare_one_two.add_precondition(Not(Equals(left, right2)))
    compare_one_two.add_precondition(Not(Equals(right1, right2)))


    for participant in participants:
        for w_left in weights:
            for w_right1 in weights:
                for w_right2 in weights:
                    if weight_name(w_left) == weight_name(w_right1) + weight_name(w_right2):
                        compare_one_two.add_effect(
                            believed_weight(left, w_left, participant),
                            True,
                            condition=And(
                                believed_weight(right1, w_right1, participant),
                                believed_weight(right2, w_right2, participant),
                                actual_weight(left, w_left),
                                actual_weight(right1, w_right1),
                                actual_weight(right2, w_right2),
                                is_paying_attention(participant)
                            )
                        )

                        compare_one_two.add_effect(
                            believed_weight(right1, w_right1, participant),
                            True,
                            condition=And(
                                believed_weight(left, w_left, participant),
                                believed_weight(right2, w_right2, participant),
                                actual_weight(left, w_left),
                                actual_weight(right1, w_right1),
                                actual_weight(right2, w_right2),
                                is_paying_attention(participant)
                            )
                        )

                        compare_one_two.add_effect(
                            believed_weight(right2, w_right2, participant),
                            True,
                            condition=And(
                                believed_weight(left, w_left, participant),
                                believed_weight(right1, w_right1, participant),
                                actual_weight(left, w_left),
                                actual_weight(right1, w_right1),
                                actual_weight(right2, w_right2),
                                is_paying_attention(participant)
                            )
                        )


    compare_one_three = InstantaneousAction('compare_one_three', left=Block, right1=Block, right2=Block, right3=Block)
    left = compare_one_three.parameter('left')
    right1 = compare_one_three.parameter('right1')
    right2 = compare_one_three.parameter('right2')
    right3 = compare_one_three.parameter('right3')

    compare_one_three.add_precondition(Not(Equals(left, right1)))
    compare_one_three.add_precondition(Not(Equals(left, right2)))
    compare_one_three.add_precondition(Not(Equals(left, right3)))
    compare_one_three.add_precondition(Not(Equals(right1, right2)))
    compare_one_three.add_precondition(Not(Equals(right1, right3)))
    compare_one_three.add_precondition(Not(Equals(right2, right3)))


    for participant in participants:
        for w_left in weights:
            for w_right1 in weights:
                for w_right2 in weights:
                    for w_right3 in weights:
                        if weight_name(w_left) == weight_name(w_right1) + weight_name(w_right2) + weight_name(w_right3):
                            # Effect for learning left block weight
                            compare_one_three.add_effect(
                                believed_weight(left, w_left, participant),
                                True,
                                condition=And(
                                    believed_weight(right1, w_right1, participant),
                                    believed_weight(right2, w_right2, participant),
                                    believed_weight(right3, w_right3, participant),
                                    actual_weight(left, w_left),
                                    actual_weight(right1, w_right1),
                                    actual_weight(right2, w_right2),
                                    actual_weight(right3, w_right3),
                                    is_paying_attention(participant)
                                )
                            )

                            # Effects for learning right block weights
                            for right, w_right in [(right1, w_right1), (right2, w_right2), (right3, w_right3)]:
                                compare_one_three.add_effect(
                                    believed_weight(right, w_right, participant),
                                    True,
                                    condition=And(
                                        believed_weight(left, w_left, participant),
                                        *[believed_weight(r, w, participant) for r, w in [(right1, w_right1), (right2, w_right2), (right3, w_right3)] if r != right],
                                        actual_weight(left, w_left),
                                        actual_weight(right1, w_right1),
                                        actual_weight(right2, w_right2),
                                        actual_weight(right3, w_right3),
                                        is_paying_attention(participant)
                                    )
                                )


    compare_two_two = InstantaneousAction('compare_two_two', left1=Block, left2=Block, right1=Block, right2=Block)
    left1 = compare_two_two.parameter('left1')
    right1 = compare_two_two.parameter('right1')
    left2 = compare_two_two.parameter('left2')
    right2 = compare_two_two.parameter('right2')

    compare_two_two.add_precondition(Not(Equals(left1, left2)))
    compare_two_two.add_precondition(Not(Equals(left1, right1)))
    compare_two_two.add_precondition(Not(Equals(left1, right2)))
    compare_two_two.add_precondition(Not(Equals(left2, right1)))
    compare_two_two.add_precondition(Not(Equals(left2, right2)))
    compare_two_two.add_precondition(Not(Equals(right1, right2)))

    for participant in participants:
        for w_left1 in weights:
            for w_left2 in weights:
                for w_right1 in weights:
                    for w_right2 in weights:
                        if weight_name(w_left1) + weight_name(w_left2) == weight_name(w_right1) + weight_name(w_right2):
                            # Effects for learning left block weights
                            for left, w_left in [(left1, w_left1), (left2, w_left2)]:
                                compare_two_two.add_effect(
                                    believed_weight(left, w_left, participant),
                                    True,
                                    condition=And(
                                        believed_weight(right1, w_right1, participant),
                                        believed_weight(right2, w_right2, participant),
                                        believed_weight(left1 if left == left2 else left2, w_left1 if left == left2 else w_left2, participant),
                                        actual_weight(left1, w_left1),
                                        actual_weight(left2, w_left2),
                                        actual_weight(right1, w_right1),
                                        actual_weight(right2, w_right2),
                                        is_paying_attention(participant)
                                    )
                                )

                            # Effects for learning right block weights
                            for right, w_right in [(right1, w_right1), (right2, w_right2)]:
                                compare_two_two.add_effect(
                                    believed_weight(right, w_right, participant),
                                    True,
                                    condition=And(
                                        believed_weight(left1, w_left1, participant),
                                        believed_weight(left2, w_left2, participant),
                                        believed_weight(right1 if right == right2 else right2, w_right1 if right == right2 else w_right2, participant),
                                        actual_weight(left1, w_left1),
                                        actual_weight(left2, w_left2),
                                        actual_weight(right1, w_right1),
                                        actual_weight(right2, w_right2),
                                        is_paying_attention(participant)
                                    )
                                )

    block1 = Object('red block', Block)
    block2 = Object('blue block', Block)
    block3 = Object('green block', Block)
    block4 = Object('purple block', Block)
    block5 = Object('yellow block', Block)
    # block6 = Object('brown block', Block)
    blocks = [block1, block2, block3, block4, block5]

    problem.add_fluents([actual_weight])
    problem.add_fluent(believed_weight, default_initial_value=False)
    problem.add_fluent(heavier, default_initial_value=False)
    problem.add_fluent(is_paying_attention, default_initial_value=False)
    problem.add_actions([set_weight, pay_attention, stop_paying_attention, compare, compare_one_two])
    #problem.add_action(compare_one_three)
    #problem.add_action(compare_two_two)
    problem.add_objects(blocks + participants)


    problem.set_initial_value(actual_weight(block1, weights[0]), True)

    for participant in participants:
        problem.set_initial_value(believed_weight(block1, weights[0], participant), True)

        for w in weights:
            statements = []
            for b in blocks:
                statement = believed_weight(b, w, participant)
                statement = And(statement, *[Not(believed_weight(b, w1, participant)) for w1 in weights if w1 != w])
            statements.append(statement)

        g = XOr(*statements)
        problem.add_goal(g)
    

    return problem, actual_weight, believed_weight, blocks, participants, weights

def create_planner():

    problem, actual_weight, believed_weight, blocks, participants, weights = create_problem()
    planner = OneshotPlanner(name="fast-downward")
    try:
        os.remove("mmdemo/features/planner/benchmarks/domain.pddl")
    except:
        print("didn't remove domain pddl with init")
        print(os.listdir("mmdemo/features/planner/benchmarks"))

    try:
        os.remove("mmdemo/features/planner/benchmarks/problem.pddl")
    except:
        print("didn't remove problem pddl with init")

    pddl_writer = PDDLWriter(problem)
    domain_file = "mmdemo/features/planner/benchmarks/domain.pddl"
    problem_file = "mmdemo/features/planner/benchmarks/problem.pddl"
    
    with open(domain_file, "w") as f:
        f.write(pddl_writer.get_domain())
    with open(problem_file, "w") as f:
        f.write(pddl_writer.get_problem())
    
    return problem, planner, actual_weight, believed_weight, blocks, participants, weights
    
def update_block_weight(self, block_name, weight_name):

    for block in self.blocks:
        if block_name in block.name :
            break
        
    found_weight = False
    for weight in self.weights:
        if weight_name in weight.name :
            found_weight = True
            break

    if found_weight:
        self.problem.set_initial_value(self.actual_weight(block, weight), True)
        for participant in self.participants:
            self.problem.set_initial_value(self.believed_weight(block, weight, participant), True)
    try:
        os.remove("mmdemo/features/planner/benchmarks/domain.pddl")
    except:
        print("didn't remove domain pddl with update")
    try:
        os.remove("mmdemo/features/planner/benchmarks/problem.pddl")
    except:
        print("didn't remove problem pddl with update")

    pddl_writer = PDDLWriter(self.problem)
    domain_file = "mmdemo/features/planner/benchmarks/domain.pddl"
    problem_file = "mmdemo/features/planner/benchmarks/problem.pddl"
    
    with open(domain_file, "w") as f:
        f.write(pddl_writer.get_domain())
    with open(problem_file, "w") as f:
        f.write(pddl_writer.get_problem())


def check_solution():

    docker_command = [
    "docker", "run", "--rm",
    "-v", "C:\\Users\\benkh\\Documents\\GitHub\\TRACE\\mmdemo\\features\\planner\\benchmarks:/benchmarks",
    "aibasel/downward", "--alias", "lama-first", "/benchmarks/problem.pddl"
    ]
    result = subprocess.run(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if "Solution found" in result.stdout:
        return True, result.stdout
    else:
        return False, result.stdout
    
problem, planner, actual_weight, believed_weight, blocks, participants, weights = create_planner()

# solv, output = check_solution()
# if solv:
#     print("Found a solution")
#     print(output)
# else:
#     print("No solution found!")