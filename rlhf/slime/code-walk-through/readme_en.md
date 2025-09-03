# A Brief Code Walkthrough of slime

slime is an extremely elegant and concise RL framework that has made tremendous optimizations in both usability and performance. Based on SGLang and Megatron LM as the only backends, slime provides strong support for MOE model training and extremely flexible sampling logic.

On the occasion of slime's 0.1.0 release, we'll quickly learn the core code of slime represented by partial rollout in this document, specifically based on commit [261ecee](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7).

## Acknowledgments

Mao Cheng @ Meta, Zhuoran Yin @ CMU, Ji Li @ Ant Group, Yixuan Zhang @ UoA, Yusheng Su @ AMD, Zhuohao Li @ Alibaba, Yuzhen Zhou @ CMU, Jiajun Li @ CMU, Biao He @ LinkedIn, Huapeng Zhou @ UW, Chengxi Li @ CMU, Chengxing Xie @Zhipu, Zilin Zhu @ Zhipu, Chenyang Zhao @ LMSYS

## Core Architecture

slime adopts a decoupled architecture, decomposing the RLHF training process into three independent collaborative modules:

- **Training (Megatron)**: Responsible for the main training process, supporting various parallel strategies; specific implementation is in [`slime/backends/megatron_utils/`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/backends/megatron_utils/).
  
- **Rollout (SGLang)**: Generates new data (including reward/verifier), based on SGLang's sampling logic; specific implementation is in [`slime/ray/rollout.py`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/ray/rollout.py).
  
- **Data Buffer**: Manages data flow and custom generation logic, which can be said to be slime's most ingenious module; specific implementation is in [`slime/ray/buffer.py`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/ray/buffer.py).

<div style="text-align: center;">
  <img src="./overall_workflow.png" alt="Overall Workflow" style="width:50%;">
</div>

Based on this forward-thinking design, slime's freedom and flexibility are incredibly refreshing:

1. **Resource Scheduling Freedom**: Supports both co-locate and dis-aggregate deployment strategies; supports DP/TP/PP/EP on rollout and training respectively; specific implementation see [`slime/ray/placement_group.py`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/ray/placement_group.py)

2. **Training Method Freedom**: Supports both synchronous and asynchronous training modes; specific implementation see [`slime/train.py`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/train.py) and [`slime/train_async.py`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/train_async.py); note that the latter requires dis-aggregate architecture, using `rollout_manager.async_generate` and `actor_model.async_train` for separated asynchronous training, where rollout always leads training by one step, i.e., one-step off-policy;

3. **Sampling Method Freedom**: Supports user-defined complex sampling processes, including [multi-turn tool calls](https://github.com/THUDM/slime/tree/main/examples/search-r1), reward model integration, custom verifiers, etc.; specific implementation see [`slime_plugins/rollout_buffer/`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime_plugins/rollout_buffer/).

4. **Model Support Freedom**: Supports both Dense and MoE models; specific scripts can refer to [`slime/scripts/run-qwen3-4B.sh`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/scripts/run-qwen3-4B.sh) and [`slime/scripts/run-deepseek-r1.sh`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/scripts/run-deepseek-r1.sh).

## Code Structure

```bash
slime/
├── slime/                          # Core framework code
│   ├── ray/                        # Ray distributed components
│   │   ├── actor_group.py          # Training Actor management
│   │   ├── rollout.py              # Inference Actor management
│   │   ├── buffer.py               # Data buffer
│   │   └── placement_group.py      # Resource allocation
│   ├── backends/                   # Backend engine integration
│   │   ├── megatron_utils/         # Megatron training backend
│   │   └── sglang_utils/           # SGLang inference backend
│   └── utils/                      # Utility functions
├── slime_plugins/                  # Plugins and extensions
│   ├── rollout_buffer/             # Custom generation plugins
│   └── models/                     # Model adapters
├── scripts/                        # Reference startup scripts
│   └── models/                     # Various model configurations
├── examples/                       # Reference usage examples
├── docs/                           # Detailed documentation
├── train.py                        # Synchronous training entry
└── train_async.py                  # Asynchronous training entry
```

Specifically:

- `scripts/`: Used to start Ray cluster and submit training jobs; example scripts choose `train.py` or `train_async.py`, such as: `slime/scripts/run-qwen3-4B.sh`, `slime/scripts/run-deepseek-r1.sh`.
- `train.py` / `train_async.py`: Training entry points, create `PlacementGroup` to allocate GPUs → create `actor_group` (training) and `rollout_manager` (inference) → enter training loop, synchronous mode executes step by step; asynchronous mode interleaves through `rollout_manager.async_generate()` and `ray.get()` for parallelization.
- `slime/ray/`: Distributed orchestration and resource management, specifically including: `placement_group.py`: GPU resource allocation and packaging based on Ray Placement Group, `actor_group.py`: Training Actor group management, exposing interfaces like `async_init/async_train/async_update_weights`, `rollout.py`: Rollout Actor (SGLang engine container), inference service routing, weight reception, `buffer.py`: Data buffering, sample batch organization, intermediate bridge between Rollout/Training.
- `slime/backends/`: Backend engine adaptation, specifically including: `megatron_utils/`: Training backend (optimizer, weight updates, distributed communication integration), `sglang_utils/`: Inference backend (SGLang wrapper, batch generation, engine lifecycle management).
- `slime_plugins/`: Pluggable extensions, specifically including: `rollout_buffer/`: Custom trajectory generator system through external linkage like HTTP/OpenAI interfaces; `models/`: Small adaptation layers for different model families.
- `examples/`: Some examples reproducing other work, such as `examples/search-r1/` demonstrating multi-turn dialogue + tool call generation and training concatenation methods.
- `docs/`: Documentation and usage guides, including model usage, SFT, AMD platform adaptation and optimization manuals.

### Concatenation Relationship

1. **Script Layer** (`scripts/`): Start Ray → Submit job → Choose `train.py` or `train_async.py` and pass parameters

2. **Entry Layer** (`train*.py`): `create_placement_groups(args)` allocate/map GPUs; `create_actor_group(args, pgs["actor"])` build training Actor group; `create_rollout_manager(args, pgs["rollout"])` build inference and data generation manager

3. **Execution Layer** (`ray/` + `backends/`): Training: `actor_group.async_train(...)` → Megatron optimization/gradient computation; Generation: `rollout_manager.async_generate(...)` → SGLang batch inference; Synchronization: `actor_group.async_update_weights()` → Push training weights to inference engine

4. **Data Flow** (`buffer.py` + plugins): `Buffer` responsible for sampling/batching/calling custom generation (`slime_plugins/rollout_buffer/`) → return training-usable samples

Note that although the execution layer functions are all decorated with `async`, both synchronous and asynchronous training use the same set of `async_train`, `async_generate`, and `async_update_weights` interfaces. The difference between synchronous and asynchronous training lies in the timing of `ray.get()` calls. Through the above chain, slime naturally connects script → entry → distributed execution → data/weight flow, achieving efficient and scalable RL post-training.

Next, let's dive into the specific code of each part, analyzing the important functions in the architecture diagram one by one.

## Ray Placement Group

This section details how slime performs GPU resource orchestration on Ray: how to create and reorder Placement Groups (PG) to achieve stable GPU ordering, how training Actors and Rollout Engines are scheduled on PG, and two deployment forms: colocate and dis-aggregate. For convenience of description, let's introduce some core concepts:

1. [`Ray Placement Group`](https://github.com/THUDM/slime/blob/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/ray/placement_group.py): Reserves a group of bundles (each containing 1 GPU + 1 CPU) in the cluster, and subsequently fixes actors to these bundles, achieving controllable and stable resource placement.

2. [`RayTrainGroup`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/ray/actor_group.py): Training-side actor manager; creates training actor handlers for each rank through [`_allocate_gpus_for_actor`](https://github.com/THUDM/slime/blob/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/ray/actor_group.py#L50), obtaining `self._actor_handlers`, then concurrently performs init/train/eval/save/update/offload on each rank.

3. [`RolloutManager`](https://github.com/THUDM/slime/tree/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/ray/rollout.py): Inference/data orchestrator, responsible for creating Rollout Engine, Data Buffer, Lock and Router; this content will be explained in subsequent sections.

### Entry Function

The entry is located in the [`create_placement_groups`](https://github.com/THUDM/slime/blob/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/ray/placement_group.py#L71) function of `Ray Placement Group`:

- Calculate the total number of GPUs `num_gpus` required for this training.
- Create a PG containing `num_gpus` bundles, each bundle requiring `{"GPU": 1, "CPU": 1}`.
- Obtain the reordered bundle index list for ensuring stable cross-node/GPU ordering.
- Divide the PG indices between training Actors and Rollout according to `rollout_offset`.

<details> <summary> create_placement_groups specific implementation </summary>

```python
def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node

    print(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]

    return {
        "actor": (pg, actor_pg_reordered_bundle_indices),
        "rollout": (pg, rollout_pg_reordered_bundle_indices),
    }
```
</details>

### Bundle Reordering

After creating the PG, slime uses a temporary `InfoActor` to get the actual `(Node IP, GPU ID)` allocated to each bundle, then reorders by node IP and GPU ID:

1. First try to parse `node_identifier` as an IPv4 address, convert it to 4 integers and sort accordingly;
2. If not an IP, try DNS resolution; if that fails, fall back to sorting by ASCII sequence of hostname characters;
3. Within the same node, sort by `gpu_id` in ascending order.

This achieves stable bundle ordering across multiple machines, avoiding scheduling mismatches due to unstable mapping.

<details> <summary> InfoActor and sort_key specific implementation</summary>

```python
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]

def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)
```
</details>

### Colocate vs Dis-aggregate

Familiar content:

- **Colocate**: Training Actors and Rollout engines alternately share the same batch of GPU resources, where `num_gpus = actor_num_nodes * actor_num_gpus_per_node`, `rollout_offset = 0`. Rollout and Actor completely share bundles; suitable for small-scale training.

- **Dis-aggregate**: Training Actors and Rollout engines use their own independent GPU pools (e.g., training takes 6 cards, rollout takes 2 cards), where `num_gpus = actor_num_nodes * actor_num_gpus_per_node + rollout_num_gpus`, `rollout_offset = actor_num_nodes * actor_num_gpus_per_node`. Rollout and Actor use different bundles; suitable for large-scale training; in this case, async-train can be performed.

### Port Allocation and Multi-machine Consistency

In multi-node/multi-card scenarios, `create_rollout_engines` will find a continuous range of available ports on the target node through `RayActor._get_current_node_ip_and_free_port`, and spread Node 0's `dist_init_addr` to other nodes of the same engine to ensure cross-machine process group consistency.

<details> <summary> RayActor._get_current_node_ip_and_free_port specific implementation</summary>

```python
def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
    address = ray._private.services.get_node_ip_address()
    address = address.strip("[]")
    port = start_port
    while not all(is_port_available(port + i) for i in range(consecutive)):
        port += 1
    return address, port
```
</details>

## Data Source with/without Buffer

This part should be the most enjoyable, as most algorithm practitioners should be able to modify the data buffer to freely use slime. `slime/ray/rollout_data_source.py` is the data source management module of the rollout system, responsible for providing training data for the rollout engine. This file defines two core classes: `RolloutDataSource` and `RolloutDataSourceWithBuffer`.

The diagram below is extremely clear, introducing the entire data acquisition process. Data Source can be `RolloutDataSource` or `RolloutDataSourceWithBuffer`.

<div style="text-align: center;">
  <img src="./datasource.png" alt="DataSource" style="width:50%;">
</div>

### RolloutDataSource

1. **Initialization**:

```python
class RolloutDataSource:
    def __init__(self, args):
        self.args = args
        self.epoch_id = 0          # Current epoch ID
        self.sample_index = 0      # Global sample index
        self.sample_offset = 0     # Offset in current epoch
        self.metadata = {}         # Metadata storage
        self.dataset = None        # Dataset object
```

Let's look at the specific initialization logic. Note that by default, `rollout_global_dataset=True`, at which point it loads the real dataset based on the `--prompt-data` in the startup parameters, otherwise `dataset=None`, and you can maintain the dataset yourself.

<details>
<summary>Initialization logic</summary>

```python
class RolloutDataSource:
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None
```
</details>

2. **`get_samples()`** and **`Sample`**:

Get a specified number of samples from the dataset:

1. Automatically handles epoch boundaries, supports dataset shuffle;
2. Generates `n_samples_per_prompt` samples for each prompt, used for GRPO;
3. Maintains `sample_offset`, `epoch_id`, `sample_index`;
4. Uses deep copy to avoid data pollution;
5. Returns samples in `list[list[Sample]]` format;

<details>
<summary>Sample class implementation and get_samples method</summary>

```python
class Sample:
    """The sample generated"""

    index: Optional[int] = None
    # prompt
    prompt: Union[str, list[dict[str, str]]] = ""
    tokens: list[int] = field(default_factory=list)
    # response
    response: str = ""
    response_length: int = 0
    label: Optional[str] = None
    reward: Optional[Union[float, dict[str, Any]]] = None
    loss_mask: Optional[list[int]] = None

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict):
        data["status"] = Sample.Status(data["status"])
        return Sample(**data)

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]
```

```python
def get_samples(self, num_samples):
    samples = []
    
    if self.dataset is not None:
        # Branch 1: Use real dataset
        if self.sample_offset + num_samples <= len(self.dataset):
            # Case 1: Current epoch has enough data
            prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
            self.sample_offset += num_samples
        else:
            # Case 2: Current epoch has insufficient data, need to enter next epoch
            prompt_samples = self.dataset.samples[self.sample_offset :]  # Take remaining data from current epoch
            num_samples -= len(prompt_samples)
            self.epoch_id += 1  # Enter next epoch
            
            # Re-shuffle dataset
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            
            # Take remaining required data from new epoch
            prompt_samples += self.dataset.samples[:num_samples]
            self.sample_offset = num_samples
        
        # Create multiple samples for each prompt (n_samples_per_prompt)
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)  # Deep copy to avoid modifying original data, and maintain index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            samples.append(group)
    else:
        # Branch 2: Don't use real dataset, create empty samples
        for _ in range(num_samples):
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = Sample(index=self.sample_index)
                self.sample_index += 1
                group.append(sample)
            samples.append(group)
    
    return samples
```
</details>

3. **`save()`** and **`load()`**:

Save `RolloutDataSource` state to file; load is used after training interruption to load state from file, ensuring data order consistency.

<details>
<summary>save and load methods</summary>

```python
def save(self, rollout_id):
    if not self.args.rollout_global_dataset:
        return  # Don't need to save when not using real dataset
    
    state_dict = {
        "sample_offset": self.sample_offset,  # Offset in current epoch
        "epoch_id": self.epoch_id,            # Current epoch ID
        "sample_index": self.sample_index,    # Global sample index
        "metadata": self.metadata,            # Metadata
    }
    
    # Save to specified path
    path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
```

```python
def load(self, rollout_id=None):
    if not self.args.rollout_global_dataset:
        return  # Don't need to load when not using real dataset
    
    if self.args.load is None:
        return  # Don't need to load when no load path specified
    
    path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
    if not os.path.exists(path):
        print(f"Checkpoint {path} does not exist.")
        return
    
    # Load state
    state_dict = torch.load(path)
    self.sample_offset = state_dict.get("sample_offset", 0)
    self.epoch_id = state_dict.get("epoch_id", 0)
    self.sample_index = state_dict.get("sample_index", 0)
    self.metadata = state_dict.get("metadata", {})
    
    # Re-shuffle dataset
    if self.args.rollout_global_dataset and self.args.rollout_shuffle:
        self.dataset.shuffle(self.epoch_id)
```
</details>

### RolloutDataSourceWithBuffer

Data class with buffer, inherits from `RolloutDataSource`, adds data buffering functionality, supports data reuse strategies designed for partial rollout.

1. **Initialization**: Completely inherits `RolloutDataSource` initialization logic, and additionally initializes `buffer_filter` method and empty `buffer` list.

<details>
<summary>RolloutDataSourceWithBuffer initialization</summary>

```python
class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []  # Data buffer
        
        # Set buffer filter
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first  # Default: first in first out
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)  # Custom filter
```
</details>

2. **`get_samples()`**:

Written very clearly, prioritizes getting data from buffer, supplements from original dataset when buffer is insufficient.

<details>
<summary>get_samples method</summary>

```python
def get_samples(self, num_samples: int) -> list[list[Sample]]:
    # 1. First get sample groups from buffer
    samples = self._get_samples_from_buffer(num_samples)
    num_samples -= len(samples)
    
    # 2. If buffer is insufficient, get remaining sample groups from original dataset
    if num_samples > 0:
        samples += super().get_samples(num_samples=num_samples)
    
    return samples
```

```python
def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
    if len(self.buffer) == 0 or num_samples == 0:
        return []  # Buffer is empty or no samples needed
    
    # Use buffer filter to get sample groups
    samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
    return samples
```

</details>

3. **`add_samples()`**:

Add sample groups to buffer. Note that `RolloutDataSource` doesn't support adding samples; here, sample groups are added, meaning partial rollout writes all requests of an entire prompt to buffer simultaneously, avoiding situations where different requests of the same prompt are used for training at different steps.

<details>
<summary>add_samples method</summary>

```python
def add_samples(self, samples: list[list[Sample]]):
    if not samples:
        return
    
    # Validate input format, ensure input is list[list[Sample]] format
    assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
    assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
    
    # Validate size of each group
    for i in range(0, len(samples)):
        assert (
            len(samples[i]) == self.args.n_samples_per_prompt
        ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
        group = samples[i]
        self.buffer.append(group)  # Add to buffer
```
</details>

4. **`pop_first()`**:

Default buffer filter, implements first-in-first-out (FIFO) data acquisition strategy.

<details>
<summary>pop_first method</summary>

```python
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)  # Take the smaller of buffer length and demand
    samples = buffer[:num_to_pop]               # Get first num_to_pop samples
    del buffer[:num_to_pop]                     # Delete these samples from buffer
    return samples
```
</details>

5. **Final data call chain**:

```bash
RolloutController.generate()
    ↓
RolloutDataSourceWithBuffer.get_samples()
    ↓
_get_samples_from_buffer() + super().get_samples()
    ↓
Return list[list[Sample]]
```

6. **Custom Buffer Filter**:

<details>
<summary>Custom Buffer Filter example</summary>

```python
# Define custom filter
def custom_buffer_filter(args, rollout_id, buffer, num_samples):
    # Sort by reward, take samples with highest reward
    sorted_buffer = sorted(buffer, key=lambda x: x[0].reward, reverse=True)
    return sorted_buffer[:num_samples]

# Set in args
args.buffer_filter_path = "path.to.custom_buffer_filter"
```
</details>

Written very clearly, very easy to extend.

## Rollout Control

Rollout is mainly controlled by two classes:

- `slime/ray/rollout.py`: `class RolloutManager` manages the lifecycle of rollout engines and router;
- `slime/ray/buffer.py`: `class RolloutController` processes rollout-generated data and converts it to training data;

<div style="text-align: center;">
  <img src="./rollout_parts.png" alt="Rollout Parts" style="width:50%;">
</div>

### RolloutManager

RolloutManager is the main controller of the rollout system, responsible for coordinating interactions between Router, Controller, and Engines.

1. **Initialization**: Initializes Router, Controller, Engines pool, and creates locks;

<details>
<summary>RolloutManager initialization</summary>

```python
class RolloutManager:
    def __init__(self, args, pg, wandb_run_id):
        self.args = args
        
        # 1. Start Router
        _start_router(args)
        
        # 2. Create Controller
        self.controller = RolloutController.options(
            num_cpus=1,
            num_gpus=0,
        ).remote(args, wandb_run_id=wandb_run_id)

        # 3. Create Engines pool
        self.all_rollout_engines = create_rollout_engines(args, pg)
        
        # 4. Multi-node configuration: if sglang engine needs to span multiple nodes, only send requests to node-0 of each engine
        nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.rollout_num_gpus_per_node)
        self.rollout_engines = self.all_rollout_engines[::nodes_per_engine]
        
        # 5. Create lock
        # Training process needs to broadcast new weights to all rollout engines
        # Meanwhile rollout engines may be processing inference requests
        # If broadcasting and inference happen simultaneously, communication deadlock may occur
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
        ).remote()
```
</details>

2. **`async_generate()`, `async_eval()`, `async_offload()`, `async_onload()`**:

These four functions all directly call down to the corresponding functions of Controller or Engines, which will be explained later.

3. **`create_rollout_engines`**:

Create SGLang engines:

<details>
<summary>create_rollout_engines implementation</summary>

```python
def create_rollout_engines(args, pg):
    if args.debug_train_only:
        return []

    # Calculate engine configuration
    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.rollout_num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine

    # Create Ray Actor
    RolloutRayActor = ray.remote(SGLangEngine)
    
    rollout_engines = []
    for i in range(num_engines):
        num_gpus = 0.2
        num_cpus = num_gpus

        # Set scheduling strategy
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        # Create engine
        rollout_engines.append(
            RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}},
            ).remote(args, rank=i)
        )

    # Port allocation and initialization
    # ... port allocation logic ...
    
    # Initialize all engines
    init_handles = [engine.init.remote(**ports) for engine, ports in zip(rollout_engines, addr_and_ports)]
    # Wait for all engines to complete initialization
    ray.get(init_handles)

    return rollout_engines
```
</details>

4. **`_start_router`**:

Start SGLang router, provide load balancing services:

<details>
<summary>_start_router implementation</summary>

```python
def _start_router(args):
    if args.sglang_router_ip is not None:
        return  # Already have external Router

    from sglang_router.launch_router import RouterArgs

    # Automatically allocate IP and port
    args.sglang_router_ip = get_host_info()[1]
    args.sglang_router_port = find_available_port(random.randint(3000, 4000))

    # Configure Router parameters
    router_args = RouterArgs(
        host=args.sglang_router_ip,
        port=args.sglang_router_port,
        balance_abs_threshold=0,
    )

    # Set log level and timeout
    if hasattr(router_args, "log_level"):
        router_args.log_level = "warn"
    if hasattr(router_args, "request_timeout_secs"):
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    # Start Router process
    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True
    process.start()
    
    # Wait for startup to complete
    time.sleep(3)
    assert process.is_alive()
```
</details>

Note that for sgl router, we can start router and engine simultaneously, but in slime they are started separately first, then engine registers with router through `add_worker`.

### RolloutController

RolloutController is the true executor of the rollout system, responsible for data generation, conversion, and management.

1. **Initialization**: Creates data source, dynamically loads rollout functions.

<details>
<summary>RolloutController initialization</summary>

```python
@ray.remote
class RolloutController:
    def __init__(self, args, wandb_run_id):
        self.args = args
        init_wandb_secondary(args, wandb_run_id)

        # Create data source
        self.data_source = RolloutDataSourceWithBuffer(args)

        # Dynamically load rollout functions
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")
```
</details>

2. **`generate()`**:

Calls rollout function for sampling then converts to training data format:

<details>
<summary>generate method implementation</summary>

```python
def generate(self, rollout_id):
    self.rollout_id = rollout_id

    # 1. Debug mode: load data from disk
    if self.args.load_debug_rollout_data:
        data = torch.load(
            open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
        )["samples"]
        data = [Sample.from_dict(sample) for sample in data]
    else:
        # 2. Normal mode: call rollout function to generate data
        data = self.generate_rollout(self.args, rollout_id, self.data_source, evaluation=False)
        
        # 3. Flatten data (if nested list)
        if isinstance(data[0], list):
            data = sum(data, [])

    # 4. Optional: save debug data
    if (path_template := self.args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=self.rollout_id))
        print(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=self.rollout_id,
                samples=[sample.to_dict() for sample in data],
            ),
            path,
        )
    
    # 5. Convert to training data format
    data = self._convert_samples_to_train_data(data)
    
    # 6. Wrap and return
    return Box(ray.put(data))
```
</details>

3. **`eval()`**:

Calls eval rollout function for sampling then performs scoring:

<details>
<summary>eval method implementation</summary>

```python
def eval(self, rollout_id):
    if self.args.debug_train_only:
        return  # Debug mode doesn't generate evaluation data

    # Call evaluation rollout function
    data = self.eval_generate_rollout(self.args, rollout_id, self.data_source, evaluation=True)
    
    # Record evaluation data
    log_eval_data(rollout_id, self.args, data)
```
</details>

4. **`_convert_samples_to_train_data`**:

Converts generated Sample objects to dictionary format required for training:

<details>
<summary>_convert_samples_to_train_data implementation</summary>

```python
def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
    """
    Convert inference generated samples to training data.
    """
    # Basic training data
    train_data = {
        "tokens": [sample.tokens for sample in samples], # prompt + response token ids
        "response_lengths": [sample.response_length for sample in samples], # response token length
        "rewards": [sample.get_reward_value(self.args) for sample in samples], # reward values
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples], # truncation flag
        "sample_indices": [sample.index for sample in samples], # sample indices
    }

    # Handle loss mask
    loss_masks = []
    for sample in samples:
        # If no loss_mask provided, create default
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        
        # Validate loss_mask length
        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    # Handle raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # Handle round_number (for rollout buffer)
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]
    
    return train_data
```
</details>

5. **`log_eval_data`**:

Records evaluation data to wandb and console:

<details>
<summary>log_eval_data implementation</summary>

```python
def log_eval_data(rollout_id, args, data):
    log_dict = {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

    print(f"eval {rollout_id}: {log_dict}")
    
    if args.use_wandb:
        log_dict["eval/step"] = (
            rollout_id
            if not args.wandb_always_use_train_step
            else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        )
        wandb.log(log_dict)
```
</details>

### Default Rollout Functions

Note that so far, we still haven't analyzed the default rollout function and eval function. Let's look at the default rollout function here:

<details>
<summary>Default rollout function</summary>

```python
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    completed_samples, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation
    )
    data_buffer.add_samples(aborted_samples)
    return completed_samples


def generate_abortable_samples(args, rollout_id, data_source, evaluation=False):
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_source))
```
</details>

Written as concisely as ever, let's continue following the `run` function downward:

<details>
<summary>run function specific implementation</summary>

```python
def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)
```
</details>

Concise to the point of surprise. Actually, the `coro` passed to `run` is a `coroutine` (coroutine) object. In the above context, the passed `coro` is `generate_rollout_async(args, rollout_id, data_source)`. When executing `run(generate_rollout_async(args, rollout_id, data_source))`:

1. `run()` function receives the `generate_rollout_async(args, rollout_id, data_source)` coroutine object;
2. `get_async_loop()` gets or creates a background event loop thread;
3. `async_loop.run(coro)` calls the `AsyncLoopThread.run()` method;
4. `asyncio.run_coroutine_threadsafe(coro, self.loop)` submits the coroutine to the background thread's event loop;
5. `.result()` blocks waiting for the coroutine to complete execution and returns the result;

The relevant code is as follows:

<details>
<summary>Coroutine submission related logic</summary>

1. `run(coro)` function itself

```python
def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)
```

2. `get_async_loop()` creates background event loop

```python
def get_async_loop():
    global async_loop
    if async_loop is None:
        async_loop = AsyncLoopThread()  # Create a background thread to run the event loop
    return async_loop
```

3. `AsyncLoopThread` class

```python
class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()  # Create new event loop
        self._thread = threading.Thread(target=self._start_loop, daemon=True)  # Create background thread
        self._thread.start()  # Start thread

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)  # Set event loop in new thread
        self.loop.run_forever()  # Let event loop run forever

    def run(self, coro):
        # Submit coroutine to background event loop and wait for result
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()
```

</details>

## SGLang Rollout

We continue to study downward. The default [`generate_rollout_async`](https://github.com/THUDM/slime/blob/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/rollout/sglang_rollout.py#L235) is directly defined in [`sglang_rollout.py`](https://github.com/THUDM/slime/blob/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/rollout/sglang_rollout.py).

```
Router → SGLang Server 1/2 → TP0/TP1/TP2/TP3 → Sample Generation → Reward Evaluation
```

Module structure:

```
slime/rollout/
├── __init__.py
├── sglang_rollout.py      # SGLang-based asynchronous sample generation
├── sft_rollout.py         # SFT training sample processing
├── filter_hub/            # Sample filters
│   ├── dynamic_sampling_filters.py
│   └── over_sampling_filters.py
└── rm_hub/                # Reward model collection
    ├── __init__.py
    ├── deepscaler.py
    ├── f1.py
    ├── math_utils.py
    └── math_dapo_utils.py
```

Core component details:

### RL Rollout

[SGLang Rollout](https://github.com/THUDM/slime/blob/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/rollout/sglang_rollout.py) is responsible for collecting actual samples for RL training. Uses `asyncio` to implement concurrent sample generation; `GenerateState` singleton class manages global generation state; supports interruption and recovery during generation; supports batch generation and reward model evaluation.

**[`GenerateState`](https://github.com/THUDM/slime/blob/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/rollout/sglang_rollout.py#L18)**

`GenerateState` is the global generation state manager: manages generation state of `Group: List[Sample]`; controls submission of `generate_and_rm_group` tasks; maintains `semaphore`, `sampling_params`, `args`, etc.

<details>
<summary>GenerateState specific implementation</summary>

```python
class GenerateState(metaclass=SingletonMeta):
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        # This semaphore controls the maximum traffic on the router to prevent router crash
        self.semaphore = asyncio.Semaphore(args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine)
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )
        self.reset()

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]):
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # generate_and_rm_group is a GRPO group, group contains multiple requests for one prompt
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)
```

</details>

**[`generate_rollout_async`](https://github.com/THUDM/slime/blob/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/rollout/sglang_rollout.py#L235)**

`generate_rollout_async` is the main function for asynchronous sample generation, mentioned earlier, passed as a coroutine object to the `run` function. This function honestly has room for improvement:

1. Initialize `dynamic_filter` and `over_sampling_filter`. `dynamic_filter` is the strategy mentioned in DAPO, discarding entire groups with reward std of 0 from data; however, `over_sampling_filter` is actually not used by default in slime; slime will enable over sample by default (set `over_sample_batch_size` greater than `rollout_batch_size`), but won't enable `over_sampling_filter` by default; let's not look at the case where `over_sampling_filter` is enabled first, at this point `target_data_size` equals `rollout_batch_size` and is less than `over_sample_batch_size`;
2. Enter the main while loop, wait until `data` gets `target_data_size`(`rollout_batch_size`) groups before exiting;
3. Enter the loop of submitting groups to router, check if current `remaining_batch_size` is less than `target_data_size`, if less, submit `over_sample_batch_size` groups to router; note that when first entering this loop, `remaining_batch_size` is 0 because no groups have been submitted yet; so it will definitely submit `over_sample_batch_size` groups to router; then `remaining_batch_size` will add `over_sample_batch_size`;
4. After submitting groups, wait for any group to finish, meaning all requests of the entire group have finished rollout;
5. If `dynamic_filter` is enabled, apply `dynamic_filter` to completed groups; if `dynamic_filter` returns False, subtract one `remaining_batch_size`, won't add to `data`;
6. Continue adding groups to `data` until `data` gets `target_data_size` groups; or, too many groups are filtered out, `remaining_batch_size` becomes less than `target_data_size`, then need to submit another `over_sample_batch_size` groups to router;
7. Until sampling gets `target_data_size` groups in `data`, exit the main while loop;
8. Note that the number of groups we submit is at least one `over_sample_batch_size`, while `target_data_size` may be less than `over_sample_batch_size`, so need to abort remaining requests of unfinished groups;

If we enable `over_sampling_filter`, then `target_data_size` equals `over_sample_batch_size`, wait for `over_sample_batch_size` groups to complete rollout before exiting loop, may still be filtered out by `dynamic_filter` in the middle, need to continue submitting more groups to router; after loop exits, we get `over_sample_batch_size` groups, then apply `over_sampling_filter` to filter out some groups (e.g., discard groups ranked last by reward std), then use for training.

If you understand the above logic, you can look at this example. We set `over_sample_batch_size` to 6, `rollout_batch_size` to 4, enable `dynamic_filter` and `over_sampling_filter`.

The upper middle part of the image shows the first submission of `over_sample_batch_size` groups to router, 6 groups' all requests start rollout simultaneously. Later we find the middle three groups have reward std of 0, filtered out by `dynamic_filter`, at this point `remaining_batch_size` becomes 3, less than `target_data_size` (at this point equals `over_sample_batch_size = 6`), so need to submit another `over_sample_batch_size` groups to router.

At this point, notice the 6 groups in the lower middle of the image, currently 4 groups finish sampling and aren't filtered out by dynamic filter, `data` already has 2 groups from above, totaling 6 groups, reaching `target_data_size`, so exit loop, abort the 3 groups in orange that haven't finished rollout. Then enter the leftmost part of the image, apply `over_sampling_filter` to the 6 groups in `data`, filter out 2 groups, finally get 4 groups for training.

<div style="text-align: center;">
  <img src="./sampling_flow.png" alt="Sampling Flow" style="width:50%;">
</div>

<details>
<summary>generate_rollout_async function</summary>

```python
async def generate_rollout_async(args, rollout_id: int, data_source) -> list[list[Sample]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        list[list[Sample]]: a list of samples generated by the rollout, the length of the list is exactly the same as the `rollout_batch_size`
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    # dynamic filter is DAPO's strategy, if a group's reward std is 0, delete all
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    over_sampling_filter = (
        load_function(args.over_sampling_filter_path) if args.over_sampling_filter_path is not None else None
    )

    # target_data_size is the total number of valid samples to get
    # By default, over sample filter is not enabled, but over_sample_batch_size will be greater than rollout_batch_size
    # Send over_sample_batch_size requests at once, wait until rollout_batch_size(target_data_size) groups
    # return then exit loop, remaining requests will be aborted
    target_data_size = args.over_sampling_batch_size if over_sampling_filter is not None else args.rollout_batch_size

    data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        # All requests of the entire group finish rollout before returning
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            if dynamic_filter is not None and not dynamic_filter(args, group):
                # If filtered out by dynamic_filter, subtract one remaining_batch_size, won't add to data
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    print(
        f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, label: {data[-1][0].label}, reward: {data[-1][0].reward}",
        flush=True,
    )

    # Because over_sampling_batch_size is always greater than rollout_batch_size
    # If waiting for rollout_batch_size groups then exit loop
    # Need to abort remaining requests of unfinished groups
    aborted_samples = await abort(args, rollout_id)

    if over_sampling_filter is not None:
        data = over_sampling_filter(args, data)[: args.rollout_batch_size]

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0].index)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    return data, aborted_samples
```
</details>

**[`generate_and_rm_group`](https://github.com/THUDM/slime/blob/261ecee700b30429ba2cf4d4c27e3fc7ae0a12c7/slime/rollout/sglang_rollout.py#L178)**

Executes `generate_and_rm` operation for each request in the sample group.

<details>
<summary>generate_and_rm_group related implementation</summary>

1. `generate_and_rm_group` function

```python
async def generate_and_rm_group(args, group: list[Sample], sampling_params: dict, evaluation=False) -> list[Sample]:
    """Generate and reward model evaluation for sample group"""
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate all samples concurrently
    group = await asyncio.gather(
        *[generate_and_rm(args, sample, sampling_params.copy(), evaluation=evaluation) for sample in group]
    )

    # For reward models that need the entire group, evaluate here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards):
            sample.reward = reward

    return group
```

2. `generate_and_rm` function

```python
async def generate_and_rm(args, sample: Sample, sampling_params: dict, evaluation=False) -> Sample:
    """Generate and reward model evaluation for single sample"""
    # For samples with existing responses, check if completed
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # Generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        if args.custom_generate_function_path is not None:
            custom_generate_func = load_function(args.custom_generate_function_path)
            sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    if sample.status == Sample.Status.ABORTED:
        return sample

    # For reward models that need the entire group, don't evaluate here
    if args.group_rm:
        return sample

    # Evaluate reward
    sample.reward = await async_rm(args, sample)
    return sample
```

3. `abort` function

```python
async def abort(args, rollout_id: int):
    """Interrupt generation process"""
    aborted_samples = []
    state = GenerateState(args)
    state.aborted = True
    
    # Interrupt all requests
    response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
    for url in response["urls"]:
        await post(f"{url}/abort_request", {"abort_all": True})

    # Collect partially completed samples
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group = task.result()
            aborted_samples.append(group)

    return aborted_samples
```
</details>

### SFT Rollout (`sft_rollout.py`)

Module specifically for supervised fine-tuning (SFT) sample processing: uses tokenizer to tokenize samples, generates loss masks for training, calculates response part length.

<details>

```python
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    # Get samples
    samples = data_buffer.get_samples(args.rollout_batch_size)
    
    for sample in samples:
        # Generate loss mask
        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages)
        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
        
        # Set sample attributes
        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]
    
    return samples
```
</details>

### `filter_hub/`

Used to implement dynamic filtering (dynamic sampling filter) and over-sampling filtering (over sampling filter) mechanisms to ensure sample quality.

1. **dynamic sampling filters**

Filter out sample groups with reward std of 0 (delete all 0/1 sample groups)

<details>
<summary>dynamic sampling filters implementation</summary>

```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    """
    Check if the reward standard deviation of the sample group is greater than 0
    
    Args:
        args: global parameters
        samples: sample list
        **kwargs: additional parameters
    
    Returns:
        bool: returns True if standard deviation is greater than 0, otherwise False
    """
    rewards = [sample.get_reward_value(args) for sample in samples]
    return torch.tensor(rewards, dtype=torch.float).std() > 0.0
```
</details>

2. **over sampling filters**

Sort sample groups by reward standard deviation, prioritize groups with high variance, not enabled by default

<details>
<summary>over sampling filters implementation</summary>

```python
def sort_by_reward_std(args, samples: list[list[Sample]], **kwargs) -> list[list[Sample]]:
    """
    Sort sample groups by reward standard deviation
    
    Args:
        args: global parameters
        samples: sample group list
        **kwargs: additional parameters
    
    Returns:
        list[list[Sample]]: sample groups sorted by standard deviation in descending order
    """
    samples_with_std = []
    for group in samples:
        rewards = [item.reward for item in group]
        std = torch.tensor(rewards, dtype=torch.float).std()
        samples_with_std.append((group, std))
    
    # Sort by standard deviation in descending order (python sort is stable)
    samples_with_std.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in samples_with_std]
```
</details>

### Reward Model Collection (`rm_hub/`)

Evaluation mechanism for generated samples, supports various evaluation methods:

- **DeepScaler**: Rule-based reward model
- **DAPO/Math**: Mathematical problem evaluation model
- **F1**: F1 score calculation model
- **Remote RM**: Remote reward model interface

1. **`async_rm`**

Evaluates reward for single sample according to configured reward model type.

<details>
<summary>async_rm implementation</summary>

```python
async def async_rm(args, sample: Sample, **kwargs):
    """
    Asynchronously evaluate reward for single sample
    
    Args:
        args: global parameters
        sample: sample to evaluate
        **kwargs: additional parameters
    
    Returns:
        float: reward value
    """
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    rm_type = args.rm_type
    response = sample.response
    label = sample.label
    
    # Handle special prefixes
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_"):]

    # Choose reward model based on type
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    else:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
```
</details>

2. **`batched_async_rm`**

Batch evaluate rewards for multiple samples, improving evaluation efficiency.

<details>
<summary>batched_async_rm implementation</summary>

```python
async def batched_async_rm(args, samples: list[Sample], **kwargs) -> list[Union[int, float]]:
    """
    Batch asynchronously evaluate rewards for multiple samples
    
    Args:
        args: global parameters
        samples: sample list
        **kwargs: additional parameters
    
    Returns:
        list[Union[int, float]]: reward value list
    """
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
```
</details>

## Summary

slime is an elegant and powerful RL framework that achieves excellent performance through its decoupled architecture design. By separating training, rollout, and data buffer into independent modules, it provides great flexibility in resource scheduling, training methods, and sampling strategies. The framework's support for both synchronous and asynchronous training, along with its sophisticated partial rollout mechanism, makes it suitable for various RLHF training scenarios.

The code structure is clean and well-organized, with clear separation of concerns between different components. The use of Ray for distributed computing and SGLang for efficient inference provides a solid foundation for scalable training. The plugin system allows for easy extension and customization, making slime adaptable to different research and production needs. 