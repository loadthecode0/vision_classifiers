from prompt_learning.runner import PromptLearningRunner

def main():
    for method in ["coop", "cocoop", "maple"]:
        runner = PromptLearningRunner(method=method, dataset="orthonet")
        runner.run_testing()

if __name__ == "__main__":
    main()

