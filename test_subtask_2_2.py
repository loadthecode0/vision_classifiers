from clip_zero_shot.runner import ZeroShotRunner

def main():
    runner = ZeroShotRunner(dataset="pacemakers")
    runner.run()
    
if __name__ == "__main__":
    main()


