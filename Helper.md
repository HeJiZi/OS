# 问题帮助

此文档中提供了部分疑难杂症的解决思路，供参考。

# Keras && Tensorflow 

## Tensorflow 万年都下载不了

1. 多试几次
2. 换个网络，比如手机热点

## 指令 pip install tensorflow 报告找不到 TensorFlow ：Could not find a version that satisfies the requirement tensorflow

1. 检查 pip 源中是否有 TensorFlow
   
   使用 `pip search tensorflow` 指令看一下输出中是否有 TensorFlow ，正常是有的。
2. 检查 python 版本

    - python 3.7 及 3.7 以上版本不支持 TensorFlow
    - 看一下 python 是 32 位的还是 64 位（ Recommeded ）的，`python` 指令
3. 检查 pip 的版本

# Spacy

## Spacy 中文模块的安装

[github 地址](https://github.com/howl-anderson/Chinese_models_for_SpaCy)

1. 下载压缩包
   
    下载一个差不多叫 zh_core_web_sm-2.x.x.tar.gz 的东西，页面里头找一找。
2. 安装压缩包
    
    使用 cmd terminal powershell 等工具 cd 到压缩包所在的文件目录，再使用指令：
    ``` shell
    pip install zh_core_web_sm-2.x.x.tar.gz
    ```
    安装压缩包
3. 建立链接

    还是命令行，可以在任意目录下，使用指令：
    ``` shell
    python -m spacy link zh_core_web_sm zh
    ```
    运行完成后就可以使用 zh 这个别名来访问这个模型了。

### 注意事项

1. 安装中文模块前先装 Spacy

2. 在安装链接过程中，可能会出现文件权限不足的问题
   
    - 指定文件的属性中修改“安全”选项卡内的文件读写权限
    - 使用管理员模式打开 cmd terminal powershell 等工具
    - 有些时候，你可能需要将压缩包换一个位置存放
