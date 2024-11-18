
# 问题背景


虽说在MindSpore\-2\.3之后的版本中不在正式的发行版中支持GPU硬件后端，但其实在开发分支版本中对GPU后端是有支持的：



![](https://img2024.cnblogs.com/blog/2277440/202411/2277440-20241118100158834-2101438825.png)

但是在安装的过程中可能会遇到一些问题或者报错，这里复现一下我的Ubuntu\-20\.04环境下的安装过程。


# Pip安装


基本的安装流程是这样的，首先使用anaconda创建一个python\-3\.9的虚拟环境，因为在MindSpore\-2\.4版本之后不再支持python\-3\.7：



```
$ conda create -n mindspore-master python=3.9

```

然后根据自己的本地环境，执行相应的pip安装指令，例如：



```
$ python3 -m pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple

```

如果pip安装期间出现超时的问题，重新执行一遍上述流程即可。安装之后，执行如下指令对安装好的MindSpore进行校验：



```
$ python -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"

```

接下来就是处理各种问题的时刻。


# version XXX not found


第一个可能出现的问题类型是各种编译工具版本不匹配的问题，例如：



```
$ python -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"
Traceback (most recent call last):
  File "", line 1, in <module>
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/__init__.py", line 18, in <module>
    from mindspore.run_check import run_check
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/run_check/__init__.py", line 17, in <module>
    from ._check_version import check_version_and_env_config
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/run_check/_check_version.py", line 28, in <module>
    from mindspore._c_expression import MSContext, ms_ctx_param
ImportError: /home/dechin/anaconda3/envs/mindspore-master/bin/../lib/libstdc++.so.6: version `CXXABI_1.3.8' not found (required by /home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/_c_expression.cpython-39-x86_64-linux-gnu.so)

```

这种情况下就是找不到`CXXABI_1.3.8`这个软件版本。但是如果检查一下系统里面的软件版本：



```
$ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep CXXABI
CXXABI_1.3
CXXABI_1.3.1
CXXABI_1.3.2
CXXABI_1.3.3
CXXABI_1.3.4
CXXABI_1.3.5
CXXABI_1.3.6
CXXABI_1.3.7
CXXABI_1.3.8
CXXABI_1.3.9
CXXABI_1.3.10
CXXABI_1.3.11
CXXABI_1.3.12
CXXABI_TM_1
CXXABI_FLOAT128

```

我们发现`CXXABI_1.3.8`是存在的，而之所以有这样的报错，是因为在anaconda创建的这个mindspore虚拟环境中不存在该版本：



```
$ strings /home/dechin/anaconda3/envs/mindspore-master/lib/libstdc++.so.6 | grep CXXABICXXABI_1.3
CXXABI_1.3.1
CXXABI_1.3.2
CXXABI_1.3.3
CXXABI_1.3.4
CXXABI_1.3.5
CXXABI_1.3.6
CXXABI_1.3.7
CXXABI_TM_1

```

那么解决的方案是这样的，我们可以直接把mindspore虚拟环境下的这个动态链接库做一个软连接，链接到系统库里面的对应动态链接库上：



```
$ ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/dechin/anaconda3/envs/mindspore-master/lib/libstdc++.so.6

```

再重新运行即可解决当前问题，类似的报错还有：



```
$ python3 -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"
Traceback (most recent call last):
  File "", line 1, in <module>
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/__init__.py", line 18, in <module>
    from mindspore.run_check import run_check
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/run_check/__init__.py", line 17, in <module>
    from ._check_version import check_version_and_env_config
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/run_check/_check_version.py", line 28, in <module>
    from mindspore._c_expression import MSContext, ms_ctx_param
ImportError: /home/dechin/anaconda3/envs/mindspore-master/bin/../lib/libgomp.so.1: version `GOMP_4.0' not found (required by /home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/lib/libmindspore_backend.so)

```

也可以用相同的方法来处理。


# cannot open shared object file


配置好上述环境之后，还有可能出现这样的报错信息：



```
$ python3 -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"
[WARNING] ME(232647,7ff51906b4c0,python3):2024-11-18-09:54:31.123.673 [mindspore/ccsrc/runtime/hardware/device_context_manager.cc:65] GetNvccRealPath] Invalid environment variable CUDA_HOME [/home], can not find nvcc file [/home/bin/nvcc], please check the CUDA_HOME.
/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/train/metrics/hausdorff_distance.py:20: UserWarning: A NumPy version >=1.22.4 and <2.3.0 is required for this version of SciPy (detected version 1.22.3)
  from scipy.ndimage import morphology
[ERROR] ME(232647:140690663584960,MainProcess):2024-11-18-09:54:32.148.524 [mindspore/run_check/_check_version.py:218] libcuda.so (need by mindspore-gpu) is not found. Please confirm that libmindspore_gpu.so is in directory:/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/run_check/../lib/plugin and the correct cuda version has been installed, you can refer to the installation guidelines: https://www.mindspore.cn/install
[ERROR] ME(232647:140690663584960,MainProcess):2024-11-18-09:54:32.148.726 [mindspore/run_check/_check_version.py:218] libcudnn.so (need by mindspore-gpu) is not found. Please confirm that libmindspore_gpu.so is in directory:/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/run_check/../lib/plugin and the correct cuda version has been installed, you can refer to the installation guidelines: https://www.mindspore.cn/install
Traceback (most recent call last):
  File "", line 1, in <module>
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/_checkparam.py", line 1367, in wrapper
    return func(*args, **kwargs)
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/context.py", line 1861, in set_context
    ctx.set_device_target(kwargs['device_target'])
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/context.py", line 495, in set_device_target
    self.set_param(ms_ctx_param.device_target, target)
  File "/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/context.py", line 187, in set_param
    self._context_handle.set_param(param, value)
RuntimeError: Unsupported device target GPU. This process only supports one of the ['CPU']. Please check whether the GPU environment is installed and configured correctly, and check whether current mindspore wheel package was built with "-e GPU". For details, please refer to "Device load error message".

----------------------------------------------------
- Device load error message:
----------------------------------------------------
Load dynamic library: libmindspore_ascend.so.2 failed. libge_runner.so: cannot open shared object file: No such file or directory
Load dynamic library: libmindspore_gpu.so.11.6 failed. libcublas.so.11: cannot open shared object file: No such file or directory
Load dynamic library: libmindspore_gpu.so.11.1 failed. libcublas.so.11: cannot open shared object file: No such file or directory
Load dynamic library: libmindspore_gpu.so.10.1 failed. libcudnn.so.7: cannot open shared object file: No such file or directory

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/core/utils/ms_context.cc:287 SetDeviceTargetFromInner

```

这里的提示是找不到`libmindspore_gpu.so.11.6`等等动态链接库的地址。那么解决的方案是这样的，我们先去系统里面搜索一下这几个库，如果有存在相应的版本号，我们把所在位置的`lib`路径配置到`LD_LIBRARY_PATH`中即可：



```
$ sudo find / -name libcublas.so*
/home/dechin/anaconda3/envs/mindspore-latest/lib/libcublas.so
/home/dechin/anaconda3/envs/mindspore-latest/lib/libcublas.so.11.3.0.106
/home/dechin/anaconda3/envs/mindspore-latest/lib/libcublas.so.11
/home/dechin/anaconda3/envs/mindsponge/lib/libcublas.so
/home/dechin/anaconda3/envs/mindsponge/lib/libcublas.so.11.3.0.106
/home/dechin/anaconda3/envs/mindsponge/lib/libcublas.so.11
/home/dechin/anaconda3/envs/mindspore-master/lib/libcublas.so
/home/dechin/anaconda3/envs/mindspore-master/lib/libcublas.so.10
/home/dechin/anaconda3/envs/mindspore-master/lib/libcublas.so.10.2.2.89
/usr/lib/x86_64-linux-gnu/libcublas.so.10.2.1.243
/usr/lib/x86_64-linux-gnu/libcublas.so.10.1.0.105
/usr/lib/x86_64-linux-gnu/stubs/libcublas.so
/usr/lib/x86_64-linux-gnu/libcublas.so
/usr/lib/x86_64-linux-gnu/libcublas.so.10

```

这里我们发现在我们新建的`mindspore-master`环境中确实没有相应的动态链接库版本，但是反而是旧版的mindspore环境下有相应的这几个动态链接库，于是我的解决方案是把旧版的mindspore环境中的`lib`配置到环境变量中，即可解决该问题：



```
$ export LD_LIBRARY_PATH=/home/dechin/anaconda3/envs/mindspore-master/lib:/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/lib:/home/dechin/anaconda3/envs/mindsponge/lib

```

再次运行测试：



```
$ python3 -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"
[WARNING] ME(232736,7f562eca06c0,python3):2024-11-18-09:55:58.717.253 [mindspore/ccsrc/runtime/hardware/device_context_manager.cc:65] GetNvccRealPath] Invalid environment variable CUDA_HOME [/home], can not find nvcc file [/home/bin/nvcc], please check the CUDA_HOME.
/home/dechin/anaconda3/envs/mindspore-master/lib/python3.9/site-packages/mindspore/train/metrics/hausdorff_distance.py:20: UserWarning: A NumPy version >=1.22.4 and <2.3.0 is required for this version of SciPy (detected version 1.22.3)
  from scipy.ndimage import morphology
MindSpore version:  2.4.0.dev20241103
The result of multiplication calculation is correct, MindSpore has been installed on platform [GPU] successfully!

```

可以看到，虽然有一些告警信息，但是最终的运行结果是正确的，需要忽略告警信息的话可以运行：



```
$ export GLOG_v=4

```

来配置mindspore日志等级。


这里有个问题是，如果用户的环境中没有安装旧版本的MindSpore。那么我个人认为比较方便的一个方案是，如果系统环境中有其他的`libcublas`，例如Jax或者Torch等框架环境下也会有这些相关的软件版本，可以把他们的所在路径直接配置到环境变量中即可。如果什么环境都没有，那我的建议是先另建一个虚拟环境，安装一个旧版本的MindSpore，例如`mindspore-gpu-2.2`，确保成功安装后，再将这个旧版的lib路径配置到新版本下的环境变量中。


# Unsupported device target GPU


如果在运行的过程中有出现`Unsupported device target GPU`的话，并且自动去索引Ascend后端的动态链接库，这种情况发生的原因是没有配置`CUDA_HOME`这个环境变量。应该是，新版本mindspore底层判断硬件平台的逻辑是通过获取环境变量来的，所以需要手动配置一个`CUDA_HOME`参数即可，例如：



```
$ export CUDA_HOME=/home

```

虽然这样随意配置有可能导致一些告警信息，但并不影响程序的正确运行结果。


# 总结概要


本文介绍了在Ubuntu\-20\.04系统下安装最新的MindSpore\-2\.4\-for\-GPU版本的方法，以及安装过程中有可能出现的一些问题。虽然在MindSpore的正式版本中已经不再支持GPU硬件后端，但是开发版本目前还是持续在支持的，并且其中包含了2\.3和2\.4版本的新特性，只是算子层面没有更新和优化。对于GPU后端的MindSpore用户来说，也算是一个好消息。


# 版权声明


本文首发链接为：[https://github.com/dechinphy/p/mindspore\-2\-4\.html](https://github.com)


作者ID：DechinPhy


更多原著文章：[https://github.com/dechinphy/](https://github.com)


请博主喝咖啡：[https://github.com/dechinphy/gallery/image/379634\.html](https://github.com):[樱花宇宙官网](https://yzygzn.com)


# 参考链接


1. [https://www.mindspore.cn/install/](https://github.com)


