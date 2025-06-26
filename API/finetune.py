import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.llm.peft.lora import LoRA
import pytorch_lightning as pl
from pathlib import Path
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from megatron.core.inference.common_inference_params import CommonInferenceParams

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=llm.llama3_8b.model(),
        source="hf://meta-llama/Meta-Llama-3-8B",
        overwrite=False,
    )

def example_dataset() -> run.Config[pl.LightningDataModule]:   
                                         #, seq_length=4096, micro_batch_size=1, global_batch_size=8, num_workers=1)
    return run.Config(llm.AlpacaDataModule, seq_length=2048, micro_batch_size=1, global_batch_size=2)

def configure_finetuning_recipe(nodes: int = 1, gpus_per_node: int = 2): #finetune model  
    return run.Partial(
        llm.finetune,
        model=llama3_8b(),
        trainer=trainer(),
        data=example_dataset(), #example_dataset()
        log=logger(),
        peft=lora(),
        optim=adam_with_cosine_annealing(),
        resume=resume()
    )

def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor: #device: number of GPU
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "TOKENIZERS_PARALLELISM": "False"
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def trainer() -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=2
    )
    trainer = run.Config(
        nl.Trainer,
        devices=2,
        max_steps=143,
        accelerator="gpu",
        strategy=strategy,
        plugins=bf16_mixed(),
        log_every_n_steps=1,
        limit_val_batches=2,
        val_check_interval=2,
        num_sanity_val_steps=0,
    )
    return trainer

def logger() -> run.Config[nl.NeMoLogger]:
    ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last=True,
        every_n_train_steps=10,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return run.Config(
        nl.NeMoLogger,
        name="nemo2_peft",
        log_dir="./results",
        use_datetime_version=False,
        ckpt=ckpt,
        wandb=None
    )

def adam_with_cosine_annealing() -> run.Config[nl.OptimizerModule]:
    opt_cfg = run.Config(
        OptimizerConfig,
        optimizer="adam",
        lr=1e-5,
        adam_beta2=0.98,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        bf16=True
    )
    return run.Config(
        nl.MegatronOptimizerModule,
        config=opt_cfg
    )

def lora() -> run.Config[nl.pytorch.callbacks.PEFT]:
    return run.Config(LoRA)

def llama3_8b() -> run.Config[pl.LightningModule]:
    return run.Config(llm.LlamaModel, config=run.Config(llm.Llama3Config8B))

def resume() -> run.Config[nl.AutoResume]:
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig,
            path="nemo://meta-llama/Meta-Llama-3-8B"
        ),
        resume_if_exists=False
    )

if __name__ == "__main__":
    # configure your function
    import_ckpt = configure_checkpoint_conversion()

    # run your experiment
    run.run(import_ckpt, executor=run.LocalExecutor())
    
    # start finetuning
    run.run(configure_finetuning_recipe(), executor=local_executor_torchrun())