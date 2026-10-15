# Add these entries to _CONFIGS in openpi src/openpi/training/config.py (after pi05_libero).

    TrainConfig(
        name="pi05_libero_lora",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
    ),

    TrainConfig(
        name="pi05_libero_lora_drawer",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="water/libero_drawer100",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            # reuse the norm stats of the full LIBERO dataset, so the model sees the same scaling as before
            assets=AssetsConfig(assets_dir="assets/pi05_libero_lora", asset_id="physical-intelligence/libero"),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        # start from your 5k-step LoRA checkpoint
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi05_libero_lora/test1/4999/params"),
        num_train_steps=2_000,
        save_interval=500,
    ),
