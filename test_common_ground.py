from featureModules.move.closure_rules import CommonGround

closure_rules = CommonGround()
class_names = ["STATEMENT", "ACCEPT", "DOUBT"]

while True:
    print()
    print()
    prop = input("enter prop: ").strip()
    cl = input("enter classes comma separated (0-statement, 1-accept, 2-doubt): ").split(",")
    cl = [int(i.strip()) for i in cl]
    cl = [("STATEMENT", "ACCEPT", "DOUBT")[i] for i in cl]

    print()
    print()
    print(f"Prop: {prop} -- {cl}")
    closure_rules.update(cl, prop)
    print("Q bank")
    print(closure_rules.qbank)
    print("E bank")
    print(closure_rules.ebank)
    print("F bank")
    print(closure_rules.fbank)
