# Docker 零帧起手

## Docker 的用途

我们经常诟病深度学习代码的不可复现性，因此，我认为学术圈的人要是真的在乎自己工作的 real impact，而不是留下一堆没法复现的代码让几年后的 reviewer 拿出来恶心人，都应该学会如何用 docker。从我上述讨论中，想必大家已经感受到了，docker 是一种将开发环境连同代码复制并分发的强大软件。在我的日常工作中，docker 起到了如下作用：

1. 提供高度可复现且隔离的开发环境：Docker 不仅确保了代码的可复现性，还提供了一个隔离的沙箱环境。在开发过程中，我们可以在容器内自由安装依赖、修改系统配置，而不会影响宿主机的环境。这种隔离性特别适合在共享集群上工作，因为即使没有 sudo 权限，我们也能在容器内完成所有必要的环境配置。同时，通过使用统一的 Docker 镜像，我们确保了所有开发者都使用完全相同的环境，**避免了"在我机器上能跑"的问题**。Talk is cheap, show me your code.
3. 便于我们在集群上 share 磁盘空间：这是一个非常现实的问题，在我们的开发集群上，磁盘空间往往是不够用的，比如 8 * H100 集群可能磁盘只有 3T，倘若大量开发者共用集群，各自建立各自的用户。每个人都会在自己的用户路径下存放自己的 huggingface cache，即便强行指定 huggingface cache 为一个统一路径，其实 huggingface 也会根据登录用户的不同来鉴权，每个用户还是独立存了自己的 cache。模型的磁盘大小都不低，而且几十号人可能都得用 Llama 3.1 8B 这种模型，磁盘当然是吃不消的。因此，我们的集群强制每个人必须使用统一的账号登录，然后建立自己的 docker，并且将外部的 huggingface cache 路径统一映射到 docker 内部，避免了 huggingface 的 cache 被反复存放，显著节省了磁盘空间。当然，给我们的开发者带来了一定不便。不过，也促进了每个人都得搞明白 docker 如何使用。当然，坏处也明显，总有人隔三差五会把别人的 docker 误删了，反过来促进了大家随时 commit and push 代码。

总之，为了共同塑造良好的科研环境，避免有人用 baseline "在我的机器上能跑"来恶心别人，学习 docker 对任何人都是必不可少的。

## 安装 docker

- 通常来讲，服务器管理员需要预先安装 docker 软件，无需你来手动安装。如果他拒绝安装 docker，建议和他据理力争。毕竟，连 docker 管理都不做，这管理员不如不要当了。
- 倘若服务器上找不到 docker，也可以使用 nerdctl 作为替代。nerdctl 完全兼容 docker 的命令。把指令替换成 `nerdctl xxx` 即可。
- 如果你是管理员，可参考 [Install Docker](https://docs.docker.com/engine/install/)。

## 下载 docker 镜像

绝大多数 docker 镜像都被发布在 [Docker Hub](https://hub.docker.com/)。为了开发 sglang，我通常使用 SGLang 的官方镜像 [lmsysorg/sglang Tags | Docker Hub](https://hub.docker.com/r/lmsysorg/sglang/tags)。当然，如果你关注 SGLang RL 小组的工作，其实我们也有专门为了 verl-SGLang 搭建的镜像，比如 jurong 的 `ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.5.post3`。你当然发现了，这个版本带有 SGLang 的版本号，并不是最新的 SGLang。因此，这个 docker 我更多用于 verl—SGLang 的 CI。实际开发 verl—SGLang，我还是会用 SGLang 本身的 docker，然后在其中安装 verl 和最新的 SGLang。

```bash
# 下载镜像
# docker pull <image-name>
docker pull lmsysorg/sglang:latest
```

## 在容器内运行 docker 镜像

下载的**镜像**相当于压缩包，我们要把镜像解成**容器**才可以运行。

运行容器的指令格式是：

```bash
docker run [OPTIONS] IMAGE [COMMAND]
```

- OPTIONS: 运行容器时附加的参数
- IMAGE: 镜像名
- COMMAND: 容器启动时运行什么指令

## 常用 OPTIONS

1. `-it` 交互式终端：有该参数你才能在 docker 里用交互式终端

```bash
# 使用 -it：可以进入容器并执行命令
docker run -it ubuntu bash
# 此时你可以输入命令，比如 ls、cd 等

# 不使用 -it：容器会立即退出
docker run ubuntu bash
# 容器会立即结束，因为无法接收输入
```

2. `--name <container-name>` 容器名，方便下次重启。
  - 标记 docker 的用途或者拥有者。
  - 命名规则请参考服务器准则。在 SGLang 的开发机器上，随意命名的容器会被直接删除。

- `--shm-size <shared-memory-size>` 共享 CPU 内存大小。一般我们做 RL 需要较大的内存，默认的 64MB 会导致崩溃，建议设置为 16g 及以上。

- `--gpus all`  允许容器 access 哪些 GPU。如无特殊需求，设置成 all 即可。

- `-v <host-path>:<container-path>` 目录挂载：这可能是最重要的功能。举个例子，前文就提到了，我们将开发集群上的统一登录用户的 huggingface cache 挂载到了每个 docker 下，避免了每个人一个 cache 满天飞。更具体的来说，`-v` 可将宿主机目录 `<host-path>` 的全部内容挂载到容器目录 `<container-path>` 。宿主机和 docker 容器会共享目录下所有的文件和文件夹。容器对内容修改对宿主机可见，反之亦然。该参数可以多次添加，通常用于映射代码工作区，数据集，模型文件，配置文件等。

我们将这些常见参数组合起来，得到如下指令：

```bash
docker run -it --name <container-name> --shm-size 16g --gpus all -v <host-path>:<container-path> IMAGE
```

## 可选 OPTIONS

1. `-p <host-port>:<container-port>` 端口映射：将容器端口 `<container-port>` 映射到宿主机端口 `<host-port>` 。使得外部可以通过宿主机端口来访问容器内运行的服务。

2. `--network host` 网络共享：使容器直接使用宿主机的网络。共享 ip，端口，网络资源等。
  - 部分服务器在国内，添加 `--network host` 以共享网络代理。
  - 使用 `--network host` 时，`-p` 参数会被忽略。

3. `-e <cv-name>=<cv-value>` 环境变量：设置容器内环境变量`<cv-name>` 的值为 `<cv-value>` ，可多次添加。

4. `--ipc=host` 进程间通信的命名空间共享：允许容器内的进程与宿主机上的进程进行通信，共享 IPC 命名空间。

5. `-d` 在后台运行容器，输入 exit 时容器不关闭。

6. `--rm` 容器关闭后自动删除。

## COMMAND

在启动 container 的同时，我们可以指定 container 立即执行的指令。譬如：

1. 在容器内启动 `bash` 终端，可以输入指令，管理文件等。

```bash
docker run -it [other OPTIONS] <image-name> bash
```

2. 在容器内启动 sglang server 服务（运行在容器的 30000 端口上），并映射到宿主机的 30000 端口，以供外部访问。同时定义环境变量 HF_TOKEN 来鉴权。

```bash
docker run -p 30000:30000 --env "HF_TOKEN=hf_xxx" [other OPTIONS] <image-name> python3 -m sglang.launch_server [other paras]
```

## 容器管理

### 容器生命周期

在 docker 中，容器的生命周期管理是最基础的操作。这里是几个核心命令的区别：

1. `docker run`：创建并启动新容器：
   - 容器不存在时：创建并启动一个新的容器，相当于"买一台新电脑并开机"
   - 容器已存在时：会报错 `Error: Conflict. The container name is already in use`，因为不能创建同名容器

2. `docker start`：启动已停止的容器：
   - 容器不存在时：报错 `Error: No such container`；
   - 容器已存在时：启动容器，保持原有配置不变，相当于"把已经关机的电脑重新开机"；

3. `docker restart`：重启容器
   - 容器不存在时：报错 `Error: No such container`；
   - 容器已存在时：先停止再启动，相当于"重启电脑"，用于 docker 崩溃时重启；

4. `docker exec`：在运行中的容器执行命令
   - 容器不存在时：报错 `Error: No such container`；
   - 容器已存在但未运行：报错 `Error: Container is not running`；
   - 容器正在运行：在容器中执行命令，相当于"在已经开机的电脑上打开一个新的终端窗口"。

### 容器操作

1. 查看容器：
   - `docker ps` 查看正在运行的容器
   - `docker ps -a` 查看所有容器（包括已停止的）
   - `docker ps -a -s` 查看所有容器及其大小

2. 关闭容器：
   - 退出会话并关闭容器：输入 `exit` 或按 Ctrl + D
   - 退出会话但保持容器运行：
     - 按 Ctrl + P, 再 Ctrl + Q（在部分 IDE 中可能失效）
     - 使用 `-d` 参数启动容器
     - 直接关闭终端

3. 删除容器：
   - 需要先停止容器：`docker stop <container-name>`
   - 然后删除容器：`docker rm <container-name>`
   - 或使用 `--rm` 参数：容器关闭后自动删除

### 使用场景举例

```bash
# 1. 首次创建并运行容器
docker run -it --name my_container ubuntu bash

# 2. 容器停止后，重新启动
docker start my_container

# 3. 容器运行中，需要重启
docker restart my_container

# 4. 容器运行中，需要打开新的终端
docker exec -it my_container bash

# 5. 如果容器已存在，想创建新容器
docker run -it --name my_container_new ubuntu bash
```

### 特别说明

- 使用 `--rm` 参数时，容器停止后会自动删除，此时 `start` 和 `restart` 都会失败
- `exec` 只能用于运行中的容器，不能用于已停止的容器
- `run` 是创建新容器，其他命令都是操作已存在的容器

## 镜像构建

TODO

## 镜像上传

TODO





