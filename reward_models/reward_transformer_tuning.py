from reward_model_transformer import train_reward_tune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

if __name__ == "__main__":
    config = {
        "lr": tune.choice([1e-5, 1e-4, 1e-3]),
        "d_model": tune.choice([64, 128, 256]),
        "epochs": 350,
        "dim_feedforward": tune.choice([128, 256, 512]),
        "nhead": tune.choice([1, 2, 4, 8]),
        "num_layers": tune.choice([1, 2, 3])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=350,
        grace_period=20,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )

    result = tune.run(
        train_reward_tune,
        resources_per_trial={"cpu": 1, "gpu": 0.1},
        config=config,
        num_samples=40,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")