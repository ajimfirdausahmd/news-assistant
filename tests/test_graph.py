import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.graph.graph_builder import build_graph

def test_query(question: str):
    graph = build_graph()
    result = graph.invoke({"question": question})
    print("\n============================")
    print("USER QUESTION:")
    print(question)
    print("============================")
    print("BOT ANSWER:")
    print(result.get("answer"))
    print("============================")
    print("STATE OUTPUT:")
    print(result)
    print("============================\n")


if __name__ == "__main__":
    test_query("What initiatives has MCMC launched?")
    test_query("How many positive and negative news are there?")
    test_query("How many news are before June 2025?")
    test_query("Adakah SSM terlibat dengan kes mahkamah?")
    test_query("What is the status of the Malaysian economy in 2025?")
