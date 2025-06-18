# 如何配置一台爽快的开发机器

前天，我的一位好朋友批评我：“你和世界上 98% 的男人都一样，见到什么美好的事物都想着占为己有，而不是欣赏这份美好”。

我觉得她说的很对，但是，我就想问，有几个男人见了这个，能把持得住？

<img src="./H200.png" width="70%" alt="H200">


总之，我是把持不住了，终于有幸摸到了金子做成的 H200。这个笔记简单记录下自己好几个月以来为了开发 SGLang for RLHF，踩坑无数次后的配环境经验。天下苦 conda 久已，我尝试完全基于原生的 python 虚拟环境管理，搭配 uv 包管理，希望能帮助读者零帧起手，配置一台爽快的开发机器。

## 配置 bash/zsh

任何一台集群起步，我都会建议先配置好 bash/zsh，最大的好处是在进行任何安装前，我们就可以确定好所有的数据路径，避免数据被写到 `/root` 或者 `/home` 等集群共用的目录下。据我所知和，大部分开发集群的数据路径都不是 `/home` 或者 `/root`，**如果向这两个路径下写入大量内容而占据所有的磁盘，会导致 `/root/tmp` 或者 `/home/tmp` 也无法写入，而 ssh 登录集群需要向这两个关键的 `tmp` 目录写入数据，所以一直往这两个目录写入数据，会导致 ssh 登录失败，集群得返厂重修。**

这里分享下我自己喜欢的一套配置，可以参考下：

<details>
<summary>我喜欢用的 .bashrc 文件，zsh 同理</summary>

```bash

## Git 相关

# 创建新的 branch
alias gcb="git checkout -b"

# 提交 commit
alias gcm="git commit --no-verify -m"

# 切换 branch
alias gc="git checkout"

# 推送本地新创建的 branch 到远端
alias gpso='git push --set-upstream origin "$(git symbolic-ref --short HEAD)"'

# 推送本地 branch 到远端
alias gp="git push"

# 查看本地 branch
alias gb="git branch"

# 拉取远端 branch
alias gpl="git pull --no-ff --no-edit"

# 添加所有文件
alias ga="git add -A"

# 设置远端 branch
alias gbst="git branch --set-upstream-to=origin/"

# 查看 commit 树
alias glg="git log --graph --oneline --decorate --abbrev-commit --all"

# 查看 commit 表格
alias gl="git log"

## python 相关

# 运行 python
alias py="python"

# 运行 pip，注意这个 pip 使用了当前 python 环境的强制对齐，可以避免很多坑
alias pip="python -m pip"

# 运行 ipython
alias ipy="ipython --TerminalInteractiveShell.shortcuts '{\"command\":\"IPython:auto_suggest.resume_hinting\", \"new_keys\": []}'"

## Devlop 需求

# 运行 pre-commit
alias pre="pre-commit run --show-diff-on-failure --color=always --all-files"

# 用人类可理解的格式查看当前路径下磁盘空间使用量
alias duh="du -hs"

# 查看磁盘空间使用量

alias dfh="df -h"

# 安装当前目录下的包
alias pi="pip install ."

# 查看当前路径下的文件
alias le="less"

# 查看历史命令
alias his="history"


# 查看当前路径下的文件树
alias tr="tree -FLCN 2"

# 查看当前路径下的文件夹
alias trd="tree -FLCNd 2"

# 每 0.1 秒读取当前路径下的磁盘空间使用量，特别在下载模型的时候很好用
alias wd="watch -n 0.1 du -hs"

# 流式读取文件的尾部内容，可以在 tmux 中提交任务然后退出 tmux，在命令行使用这个命令查看任务的 log
alias tf="tail -f"


# 设置文件权限为完全透明
alias c7="chmod 777 -R"

# 打开当前路径
alias op="open ."

# 用 cursor 打开某个文件，用 vscode 同理
alias cur="cursor"

# 用 vscode 打开某个文件
alias cod="code"

# 打开配置文件
alias ope="cursor /home/chenyang/.zshrc"

# 用 cursor 打开当前路径下的文件
alias co="cursor ."

# 快速查看 GPU 使用情况
alias nvi="nvidia-smi"

# 每 1 秒查看 GPU 使用情况，需要先 pip install gpustat
alias gpu="watch -n 1 gpustat"

# 创建 tmux 会话
alias tns="tmux new -s"

# 列出 tmux 会话
alias tls="tmux ls"

# 重新登录回到某个 tmux 会话
alias tat="tmux attach -t"

# 重新加载 zsh 配置
alias sz="source ~/.zshrc"

# 重新加载 bash 配置
alias zb="source ~/.bashrc"

# 杀死进程
alias k9="kill -9"

# 格式化代码
alias bp="black *.py && black *.ipynb"

# 杀死自己名下的所有 python 进程，慎用
alias kp="ps aux | grep '[p]ython' | awk '{print \$2}' | xargs -r kill -9"

# 删除 ipynb 文件的输出
alias nbs='find . -name "*.ipynb" -exec nbstripout {} \;'

# 用 soft方式重置 git 提交
alias grs="git reset --soft"

# 设置 huggingface 的 token

export HF_TOKEN="*******************"

# 设置 huggingface 的 cache 路径，请一定配置好，避免一个集群重复下载某个模型多次

export HF_DATASETS_CACHE="/data/.cache/huggingface/datasets"
export HF_HOME="/data/.cache/huggingface"

# 设置个人默认路径，我一般连带着所有数据一起放在 /data 下
export HOME="/data/chenyang"

# 设置 ray 的 cache 路径，如果不用 ray 不太需要管
export RAY_ROOT_DIR="/data/.cache/ray"

# 设置 wandb 的 api key
export WANDB_API_KEY="*********************"

# 设置 LD_LIBRARY_PATH，这是配置 flash attention 踩的坑，遇到问题可以参考
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# 用于分配显卡的函数，al k 可以分配 k 张空闲的卡，

function al() {
    local num_gpus=$1
    
    echo "Looking for $num_gpus free GPUs..."
    echo "Checking GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits
    
    local gpu_ids=$(nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits | 
                   awk -F, '{
                       gsub(/ /, "", $1); 
                       gsub(/ /, "", $2);
                       if ($1 + 0 < 100) print $2
                   }' | 
                   head -n $num_gpus)
    
    local found_count=$(echo "$gpu_ids" | wc -l)
    

    if [ $found_count -eq $num_gpus ]; then
        gpu_ids=$(echo "$gpu_ids" | paste -sd "," -)
        export CUDA_VISIBLE_DEVICES=$gpu_ids
        echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
        return 0
    else
        echo "Error: Requested $num_gpus GPUs but only found $found_count free GPUs"
        return 1
    fi
}

## 激活虚拟环境

function ca() {
    env_name="$1"
    # 根据环境名构造激活脚本的路径
    activation_script="$HOME/.python/${env_name}/bin/activate"
    
    # 检查激活脚本是否存在
    if [ ! -f "$activation_script" ]; then
        echo "Error: 激活脚本 '$activation_script' 不存在"
        return 1
    fi

    # 激活虚拟环境
    # 注意：使用 source 激活后，当前 shell 环境会被修改
    source "$activation_script"
    
    ceiling="===== Activated Env: ${env_name} ====="
    echo "$ceiling"
    
    # 输出当前 python 路径和版本
    python_path=$(which python)
    echo "Python 路径：$python_path"
    python --version

    # 如果你想检查环境是否切换成功，可以考虑检查环境变量
    # 比如，在虚拟环境中通常会设置 VIRTUAL_ENV 变量
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "===== NO!! Environment switch failed ====="
        return 1
    else
        echo "===== YES!! Environment switched to: $VIRTUAL_ENV ====="
    fi
}


# 读取时间戳，用于给 log 做标记

function now() {
    date '+%Y-%m-%d-%H-%M'
}

# 我个人的工作路径

alias sgl="cd /data/chenyang/sglang/python"
alias rlhf="cd /data/chenyang/OpenRLHF-SGLang/openrlhf"
alias vllm="cd /data/chenyang/vllm/"
alias docs="cd /data/chenyang/sglang/docs"
alias test="cd /data/chenyang/sglang/test"
alias awe="cd /data/chenyang/Awesome-ML-SYS-Tutorial"

# uv 相关

# 查看当前虚拟环境
alias uvv="uv venv"

## 分配 1 张 GPU
al 1

## 激活 sglang 环境
ca sglang
sleep 1
clear
```

</details>

## 安装 uv

uv 是更加现代的 python 包管理器，可以完全替代 conda。轻量级，方便，快速且强大，是未来趋势。

首先，登录崭新的集群账号，这一刻会使用全集群共用的 python 环境。这一环境在比较安全的集群管理上是不可修改的，所以我们需要开辟自己的虚拟环境来安装 uv。

```bash
# 创建虚拟环境
python3 -m venv ~/.python/sglang

# 激活虚拟环境
source ~/.python/sglang/bin/activate

# 安装 uv
pip install uv
```



## 配置 ssh

```bash
ssh-keygen
```

把公钥扔到 github 就好了。然后本地配置 git config：

```bash
git config --global user.name "zhaochenyang20"
git config --global user.email "zhaochenyang20@gmail.com"
```

## 配置 oh-my-zsh

其实我一般不自己折腾换 shell，但是如果默认就是 zsh 的话，我还是喜欢用 oh-my-zsh：

**注意 oh-my-zsh 会覆盖之前所有的 .zshrc 文件，所以需要先备份！！！**


```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```


