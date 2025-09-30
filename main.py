from finetune.runner import FinetuneRunner
from clip_zero_shot.runner import ZeroShotRunner
from prompt_learning.runner import PromptLearningRunner
from prompt_learning.hier_runner import HierarchicalPromptLearningRunner

def main():
    
    for dataset in ["orthonet", "pacemakers"]:

        # subtask 1
        for model_type in ["imagenet", "clip", "dinov2"]:
            runner = FinetuneRunner(dataset, model_type)
            runner.run_training()
            runner.run_testing()

        # subtask 2
        runner = ZeroShotRunner(dataset)
        runner.run()

        # subtask 3
        for method in ["coop", "cocoop", "maple"]:
            runner = PromptLearningRunner(method, dataset)
            runner.run_training()
            runner.run_testing()

    # subtask 3 only for pacemakers
    for method in ["coop", "cocoop", "maple"]:
        runner = HierarchicalPromptLearningRunner(method)
        runner.run_testing_stage1()

if __name__=="__main__":
    main()
    