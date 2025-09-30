from finetune.runner import FinetuneRunner

def main():
    for model_type in ["imagenet", "clip", "dinov2"]:
        runner = FinetuneRunner(dataset="pacemakers", model_type=model_type)
        runner.run_training()

if __name__ == "__main__":
    main()


