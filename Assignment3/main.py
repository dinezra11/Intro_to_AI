## Main run script
from parser import parse_yaml
from hurricane_bn import build_bn
from inference import query
from print_bn import print_bn


def print_menu():
    print("\n=== MENU ===")
    print("1) Reset evidence")
    print("2) Add evidence")
    print("3) Run probabilistic reasoning")
    print("4) Quit")


def add_evidence(evidence):
    print("\nAdd evidence:")
    print("a) Weather")
    print("b) Flooding at edge")
    print("c) Evacuees at vertex")

    choice = input("> ").strip().lower()

    if choice == "a":
        val = input("Enter weather (mild / stormy / extreme): ").strip()
        if val not in ["mild", "stormy", "extreme"]:
            print("Invalid weather value.")
            return
        evidence["W"] = val

    elif choice == "b":
        idx = input("Enter edge index: ").strip()
        val = input("Flooded? (true / false): ").strip().lower()

        if not idx.isdigit() or val not in ["true", "false"]:
            print("Invalid input.")
            return

        evidence[f"F{idx}"] = (val == "true")

    elif choice == "c":
        idx = input("Enter vertex index: ").strip()
        val = input("Evacuees present? (true / false): ").strip().lower()

        if not idx.isdigit() or val not in ["true", "false"]:
            print("Invalid input.")
            return

        evidence[f"Ev{idx}"] = (val == "true")

    else:
        print("Invalid option.")


def run_reasoning(bn, edges, n_vertices, evidence):
    print("\n=== POSTERIOR PROBABILITIES ===")

    print("\n(3) WEATHER DISTRIBUTION:")
    print(query(bn, "W", dict(evidence)))

    print("\n(2) EDGE FLOODING PROBABILITIES:")
    for i in range(len(edges)):
        print(f"F{i}:", query(bn, f"F{i}", dict(evidence)))

    print("\n(1) EVACUEE PROBABILITIES:")
    for v in range(n_vertices):
        print(f"Ev{v}:", query(bn, f"Ev{v}", dict(evidence)))


def main():
    n, edges, P1, weather_prior = parse_yaml("environment_nb_config.yaml")
    bn = build_bn(n, edges, P1, weather_prior)

    # ----------------------------
    # PART I: Print BN
    # ----------------------------
    print_bn(bn, edges, n, P1)

    # ----------------------------
    # PART II: Interactive querying
    # ----------------------------
    evidence = {}

    print("\n\n=== INTERACTIVE BAYESIAN REASONING ===")

    while True:
        print_menu()
        choice = input("> ").strip()

        if choice == "1":
            evidence.clear()
            print("Evidence reset.")

        elif choice == "2":
            add_evidence(evidence)

        elif choice == "3":
            run_reasoning(bn, edges, n, evidence)

        elif choice == "4":
            print("Goodbye.")
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
