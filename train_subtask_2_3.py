from prompt_learning.hier_runner import HierarchicalPromptLearningRunner

def main():
    for method in ["coop", "cocoop", "maple"]:
        runner = HierarchicalPromptLearningRunner(method=method)
        runner.run_stage1()
        runner.run_stage2()

if __name__ == "__main__":
    main()


