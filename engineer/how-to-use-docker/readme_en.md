## 1. What is Docker? Why Use Docker?

Docker is a software tool that packages environments into **images**. Users can run these images in **containers**, ensuring an identical environment without the need for manual configuration.

- By using the official PyTorch image, you no longer need to worry about version compatibility issues between Python, CUDA, and cuDNN when running deep learning code.

## 2. Installing Docker

Please note that the development environment for sglang is a Linux system.

- Typically, server administrators pre-install Docker, so manual installation is not required. You do not need sudo privileges to install the software.
- If Docker is not found on the server, `nerdctl` may be used as a replacement. `nerdctl` is fully compatible with Docker commands; simply replace the command with `nerdctl xxx`.
- If manual installation is necessary, refer to [Install | Docker Docs](https://docs.docker.com/engine/install/) or consult GPT.

## 3. Downloading Docker Images

Most Docker images are published on [Docker Hub](https://hub.docker.com/).

Please use the official sglang image: [lmsysorg/sglang Tags | Docker Hub](https://hub.docker.com/r/lmsysorg/sglang/tags).

```
# Download image
# docker pull <image-name>
docker pull lmsysorg/sglang:latest
```

## 4. Running Docker Images in Containers

A downloaded **image** is like a compressed package, which must be extracted into a **container** to run. The **host machine** refers to the server.

The command format for running a container is:

```
docker run [OPTIONS] IMAGE [COMMAND]
```

- **OPTIONS**: Additional parameters when running the container.
- **IMAGE**: The name of the image.
- **COMMAND**: The command executed when the container starts.
- **Notice**: The image name `IMAGE` must be placed after `OPTIONS` and before `COMMAND`.

### 4.1 Common OPTIONS (Must-Read)

- `-it` Interactive terminal
  - This option allows interactive terminal usage within Docker.
- `--name <container-name>` Container name
  - Helps identify ownership of the container.
  - Follow server naming conventions; otherwise, administrators may delete it as an orphaned container.
- `--shm-size <shared-memory-size>` Shared memory size
  - Deep learning frameworks require large memory. The default 64MB may cause crashes. It is recommended to set it to at least 16GB.
- `--gpus all` Allow access to GPUs
  - Set to `all` unless specific GPU allocation is needed.
- `-v <host-path>:<container-path>` Volume mounting
  - Mounts the entire content of `<host-path>` from the host machine to `<container-path>` in the container. Files and directories in the mounted path are shared between the host and container, with modifications reflected on both sides.
  - This option can be used multiple times, commonly for mapping workspaces, datasets, model files, and configuration files.
  - You can map the working directory to the container, run code inside the container, and develop on the host machine.
  - Example: `-v ~/.cache/huggingface:/root/.cache/huggingface` mounts the Hugging Face cache from the host, allowing the container to reuse downloaded models without redownloading.
  - On Linux, the `pwd` command outputs the current directory path. Use `-v $(pwd):<container-path>` to mount the directory where `docker run` is executed.

A complete command might look like this:

```
docker run -it --name <container-name> --shm-size 16g --gpus all -v <host-path>:<container-path> IMAGE
```

### 4.2 Optional OPTIONS

- `-p <host-port>:<container-port>` Port mapping
  - Maps `<container-port>` inside the container to `<host-port>` on the host, enabling external access.
  - Can be used multiple times.
- `--network host` Network sharing
  - Allows the container to use the host’s network directly, sharing IP addresses, ports, and network resources.
  - If the server is located in China, add `--network host` to share the network proxy.
  - When using `--network host`, the `-p` option is ignored.
- `-e <env-name>=<env-value>` Environment variables
  - Sets the container’s environment variable `<env-name>` to `<env-value>`.
  - Can be used multiple times.
- `--ipc=host` Inter-process communication namespace sharing
  - Can be used instead of `--shm-size`.
- `-d` Runs the container in the background; the container does not stop when exiting the terminal.
- `--rm` Automatically removes the container upon exit.

### 4.3 IMAGE

- **For users**: `docker pull lmsysorg/sglang:latest`
- **For developers**: `docker pull lmsysorg/sglang:dev`
- Some development issues may require different images; contact the issue owner if needed.

### 4.4 COMMAND

You can specify commands to run or restart within the container.

Some common cases:

- Start a session inside the container. The container acts like a virtual machine where you can enter commands and manage files.

```
docker run -it [other OPTIONS] <image-name> bash
```

- Start the `sglang` server inside the container (running on port 30000) and map it to port 30000 on the host for external access.
- Define the environment variable `HF_TOKEN` to specify the model token.

```
docker run -p 30000:30000 --env "HF_TOKEN=hf_xxx" [other OPTIONS] <image-name> python3 -m sglang.launch_server [other parameters]
```

## 5. Container Management

### 5.1 Viewing Containers

- `docker ps` View running containers.
- `docker ps -a` View all containers.

### 5.2 Stopping Containers

Exit the session and stop the container:

- Enter `exit`
- Or press `Ctrl + D`

Exit the session while keeping the container running in the background:

- Press `Ctrl + P`, then `Ctrl + Q`. (May not work in some IDEs due to shortcut conflicts.)
- Or add `-d` when running `docker run`; the container remains running after exiting.
- Or simply close the terminal.

Stop a running container:

- `docker stop <container-name>`

**Note**: A container automatically stops when the `COMMAND` finishes execution.

### 5.3 Restarting Containers

- `docker restart <container-name>`

### 5.4 Attaching to a Container

A container must be running before attaching.

- `docker exec -it <container-name> bash` Start an interactive session inside the container.
- The `exec` command runs a command inside a running container and can also modify environment variables.

### 5.5 Deleting a Container

The container must be stopped first:

- `docker rm <container-name>`
- Or add `--rm` when running `docker run`, so the container is automatically deleted upon exit.

## 6. Building Images

TODO

## 7. Uploading Images

TODO

## Use docker in VSCode

Please make sure to install the Devcontainer extension in VSCode.

In the folder you want to open inside the docker container create a folder `.devcontainer`.
In the `.devcontainer` create a `devcontainer.json` like so:

```
{
  "name": "simon",
  "build": {
    "dockerfile": "../docker/cuda_new.Dockerfile"
  },
  "runArgs": [
    "--gpus",
    "device=7"
  ]
}
```
Make sure to point with the path to the docker file you want to use.
You can than use 
`ctrl + shift + p` and type `Devcontainers: Open Folder in Container` and select the folder where you created the subfolder `.devcontainer`. You will be able to use all the features of VSCode while developing inside the container..

For more details see [here](https://code.visualstudio.com/docs/devcontainers/containers).