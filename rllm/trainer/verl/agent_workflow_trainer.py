import asyncio
import math
import threading
import uuid
from collections import Counter, defaultdict
from functools import reduce
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout_engine import RolloutEngine
from rllm.workflows.workflow import TerminationReason
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    _timer,
    compute_advantage,
    compute_data_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class AgentWorkflowPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        workflow_class=None,
        workflow_args=None,
    ):
        # TODO: remove after patching verl
        if config.algorithm.rejection_sample.enable:
            assert config.trainer.rejection_sample == False, (
                "Rejection sampling through config.trainer.rejection_sample is deprecated. \
                Please set config.trainer.rejection_sample = False and config.algorithm.rejection_sample.enable = True"
            )
            config.data.gen_batch_size = int(config.data.train_batch_size * config.algorithm.rejection_sample.multiplier)

        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        if self.config.algorithm.stepwise_advantage.enable:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied trajectory-wise")

    def init_workers(self):
        super().init_workers()

        assert self.config.workflow.async_engine, "Async engine is required when using a workflow."
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Async rollout mode is required when using a workflow."

        rollout_engine = RolloutEngine(
            engine_name="verl",
            tokenizer=self.tokenizer,
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            disable_thinking=self.config.workflow.disable_thinking,
        )

        self.agent_execution_engine = AgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.workflow.n_parallel_tasks,
            retry_limit=self.config.workflow.retry_limit,
        )

        asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        batch = None
        solve_none = 0
        solve_all = 0
        solve_partial = 0
        num_tasks = 0
        termination_counts = Counter()
        metrics = {}
        timing_raw = {}

        for epoch in range(self.config.trainer.total_epochs):
            # pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_tasks += len(new_batch.batch)

                new_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                new_batch = new_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                new_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])
                new_batch.meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                }

                with _timer("step", timing_raw):
                    # generate trajectories
                    final_gen_batch_output = self.generate_trajectories(batch=new_batch, timing_raw=timing_raw)

                    # need to repeat to make shape match
                    repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                    new_batch = new_batch.repeat_by_counts(repeat_counts, interleave=True)
                    final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                    new_batch = new_batch.union(final_gen_batch_output)

                    # rejection sampling (at the task level)
                    # we do rejection sampling at the task level instead of the step level
                    # in multi-step workflows, this can be overly optimistic
                    # (e.g., turn 1 all incorrect and turn 2 all correct -> keep the uid, but once we group by step both turns get have 0 advantage)
                    # but stepwise rejection sampling will not work for the broadcast mode since we cannot drop the last step and keep earlier steps
                    uids = new_batch.non_tensor_batch["task_ids"]
                    unique_uids = np.unique(uids)
                    is_correct = new_batch.non_tensor_batch["is_correct"]
                    is_valid = new_batch.non_tensor_batch["is_valid"]  # exclude items that will be filtered out later
                    drop_uids = set()

                    for uid in unique_uids:
                        candidate_rows = (uids == uid) & is_valid
                        candidate_is_correct = is_correct[candidate_rows]

                        # Check if all episodes are correct or incorrect
                        if not candidate_is_correct.any():  # also catches case where not candidate_rows.any() i.e., all rows are invalid
                            drop_uids.add(uid)
                            solve_none += 1
                        elif candidate_is_correct.all():
                            drop_uids.add(uid)
                            solve_all += 1
                        else:
                            solve_partial += 1

                    # collect and log termination reasons
                    termination_reasons = new_batch.non_tensor_batch["termination_reasons"]
                    termination_counts.update(termination_reasons)

                    # If no valid samples remain, skip this batch and get a new one
                    if len(drop_uids) == len(unique_uids):
                        print("No valid samples remain, skipping batch")
                        continue

                    if not self.config.algorithm.rejection_sample.enable:
                        batch = new_batch
                    else:
                        rejection_mask = np.isin(uids, list(drop_uids))
                        new_batch = new_batch[~rejection_mask]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        if solve_partial < self.config.data.train_batch_size:
                            continue
                        else:
                            # randomly select bsz task uids from batch, then filter batch to only contain these tasks
                            # TODO: add heuristic for selecting train_batch_size uids
                            uids = batch.non_tensor_batch["task_ids"]
                            unique_uids = np.unique(uids)
                            assert len(unique_uids) >= self.config.data.train_batch_size, "Not enough unique uids to sample from"
                            selected_uids = np.random.choice(unique_uids, size=self.config.data.train_batch_size, replace=False)
                            selected_mask = np.isin(uids, selected_uids)
                            batch = batch[selected_mask]

                    if self.config.algorithm.stepwise_advantage.enable and self.config.algorithm.stepwise_advantage.mode == "broadcast":
                        # need to make sure both number of last steps (number of uids) and number of total steps in the batch
                        # (batch size after processing) are both multiples of world size

                        # first we split the batch in two: one with only the last steps of each trajectory and the other with the remaining steps
                        is_last_step = batch.non_tensor_batch["is_last_step"]
                        valid_last_step_indices = np.where(is_last_step == True)[0]
                        not_last_step_indices = np.where(is_last_step == False)[0]
                        last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                        non_last_step_batch = batch.select_idxs(not_last_step_indices)

                        # round down last_step_batch to make sure its multiple of world size
                        num_trainer_replicas = self.actor_rollout_wg.world_size
                        max_batch_size = (last_step_batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas

                        size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                        size_mask[:max_batch_size] = True
                        last_step_batch = last_step_batch[size_mask]  # filtered last steps

                        # now we go through all the non_last_step_batch and keep everything that has same trajectory_id that exists in the filtered last steps
                        valid_last_step_trajectory_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_trajectory_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_mask = np.isin(non_last_step_trajectory_ids, valid_last_step_trajectory_ids)
                        non_last_step_batch = non_last_step_batch[non_last_step_mask]

                        # concatenate then pad
                        batch = DataProto.concat([last_step_batch, non_last_step_batch])
                        batch = self._pad_dataproto_to_world_size(batch)

                    else:
                        # then we just pad the batch size to a multiple of world size
                        batch = self._pad_dataproto_to_world_size(batch=batch)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(new_batch)
                            new_batch = new_batch.union(values)

                    with _timer("adv", timing_raw):
                        # recompute old_log_probs
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        if self.config.algorithm.stepwise_advantage.enable and self.config.algorithm.stepwise_advantage.mode == "per_step":
                            batch.batch["token_level_scores"] = batch.batch["step_rewards"]
                            batch.batch["token_level_rewards"] = batch.batch["step_rewards"]
                        else:
                            batch.batch["token_level_scores"] = batch.batch["traj_rewards"]
                            batch.batch["token_level_rewards"] = batch.batch["traj_rewards"]

                        # if we're not using computing advantages stepwise (i.e., for cumulative agents) step_ids == trajectory_ids
                        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                        if self.config.algorithm.stepwise_advantage.enable and self.config.algorithm.stepwise_advantage.mode == "broadcast":
                            is_last_step = batch.non_tensor_batch["is_last_step"]
                            last_step_indices = np.where(is_last_step == True)[0]
                            not_last_step_indices = np.where(is_last_step == False)[0]
                            non_last_step_batch = batch.select_idxs(not_last_step_indices)
                            batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            # last_step_batch contains no padded steps as it was rounded down (not padded) to a multiple of world size
                        else:
                            batch = self._remove_padding(batch)  # compute advantages over non-padded steps only

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            mask_truncated_samples=False,  # deprecated, loss is masked based on termination reason
                            clip_advantages=self.config.algorithm.clip_advantages,
                        )

                        if self.config.algorithm.stepwise_advantage.enable and self.config.algorithm.stepwise_advantage.mode == "broadcast":
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, non_last_step_batch)
                            batch = DataProto.concat([batch, non_last_step_batch])

                    # remove invalid items filtered out due to compact filtering
                    is_valid = batch.non_tensor_batch["is_valid"]
                    valid_idxs = np.where(is_valid == True)[0]
                    batch = batch.select_idxs(valid_idxs)

                    # re-pad batch size to world size for gradient update
                    batch = self._pad_dataproto_to_world_size(batch=batch)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect and logmetrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                metrics["batch/solve_none"] = solve_none / num_tasks
                metrics["batch/solve_all"] = solve_all / num_tasks
                metrics["batch/solve_partial"] = solve_partial / num_tasks

                for r in TerminationReason:
                    metrics[f"batch/{r.value}"] = termination_counts[r.value] / num_tasks

                metrics["batch/num_tasks"] = num_tasks

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                batch = None
                solve_none = 0
                solve_all = 0
                solve_partial = 0
                num_tasks = 0
                termination_counts = Counter()
                metrics = {}
                timing_raw = {}

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        is_correct_lst = []
        data_source_lst = []
        uid_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)

            test_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "agent_rollout": True,
            }

            test_output_gen_batch = self.generate_trajectories(batch=test_batch)
            repeat_counts = test_output_gen_batch.meta_info["repeat_counts"]
            # need to repeat to make shape match
            test_batch = test_batch.repeat_by_counts(repeat_counts, interleave=True)
            test_output_gen_batch.meta_info.pop("repeat_counts", None)  # no longer needed after this
            test_batch = test_batch.union(test_output_gen_batch)

            seen_episodes = set()
            selected_idxs = []
            for i, episode_id in enumerate(test_batch.non_tensor_batch["episode_ids"]):
                if episode_id not in seen_episodes:
                    seen_episodes.add(episode_id)
                    selected_idxs.append(i)
            test_batch = test_batch.select_idxs(selected_idxs)

            is_correct_lst.extend(test_batch.non_tensor_batch["is_correct"])
            uid_lst.extend(test_batch.non_tensor_batch["task_ids"])

            data_sources = test_batch.non_tensor_batch.get("data_source", None)
            if data_sources is None:
                data_sources = ["unknown"] * len(test_batch)
            data_source_lst.extend(data_sources)

        metrics = {}
        is_correct_array = np.array(is_correct_lst)
        uid_array = np.array(uid_lst)
        data_source_array = np.array(data_source_lst)

        for data_source in np.unique(data_source_array):
            pass_rates = defaultdict(list)

            data_source_mask = data_source_array == data_source
            is_correct_data_source = is_correct_array[data_source_mask]
            uids_data_source = uid_array[data_source_mask]

            for is_correct, uid in zip(is_correct_data_source, uids_data_source, strict=False):
                pass_rates[uid].append(is_correct)

            metrics[f"val/{data_source}/pass@1"] = np.mean(is_correct_data_source)
            metrics[f"val/{data_source}/pass@{n_val_samples}"] = np.mean([1 if any(pass_rate) else 0 for pass_rate in pass_rates.values()])

        return metrics

    def generate_trajectories(self, batch, timing_raw=None, **kwargs):
        """
        Generates trajectories asynchronously using the agent execution engine's excute tasks method.
        Post-processing is done in the engine as well.

        Args:
            batch: The input batch for trajectory generation
            timing_raw: Dictionary to store timing information for profiling
            **kwargs: Additional arguments to pass to trajectory_generator

        Returns:
            list: List of collected processed trajectories
        """
        if timing_raw is None:
            timing_raw = {}

        with _timer("generate_trajectories", timing_raw):
            coro = self.agent_execution_engine.execute_tasks_verl(batch, **kwargs)
            final_gen_batch_output = asyncio.run_coroutine_threadsafe(coro, self._loop).result()

        return final_gen_batch_output

    def _stepwise_advantage_broadcast(self, last_step_batch, non_last_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps within the same episode and trajectory.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants

        src_traj_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
        src_eps_ids = last_step_batch.non_tensor_batch["episode_ids"]
        src_steps = last_step_batch.non_tensor_batch["step_nums"]
        src_mask = last_step_batch.batch["response_mask"]
        src_advantages = last_step_batch.batch["advantages"]

        tgt_traj_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
        tgt_eps_ids = non_last_step_batch.non_tensor_batch["episode_ids"]
        tgt_mask = non_last_step_batch.batch["response_mask"]

        # Build id -> scalar advantage
        traj_ep_to_scalar_adv = {}
        for i, (traj_id, eps_id) in enumerate(zip(src_traj_ids, src_eps_ids, strict=False)):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.algorithm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            traj_ep_to_scalar_adv[(traj_id, eps_id)] = scalar

        # Create new tensor for non_last_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=traj_ep_to_scalar_adv[(traj_id, eps_id)], dtype=torch.float32) for i, (traj_id, eps_id) in enumerate(zip(tgt_traj_ids, tgt_eps_ids, strict=False))])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        non_last_step_batch.batch["advantages"] = final_advantage
        non_last_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        batch = self._remove_padding(batch)  # Remove any padded steps from the batch (just in case)
        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True
            batch.non_tensor_batch["is_valid"][idx] = False

        return batch

    def _remove_padding(self, batch):
        """Removes padded steps from the batch"""
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        non_pad_step_indices = np.where(is_pad_step == False)[0]
        batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
        return batch

    def shutdown(self):
        """A cleanup method to gracefully stop the background event loop."""
        self.agent_execution_engine.shutdown()
        if hasattr(self, "_loop") and self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join()
