# maml-rl

+ `futures = sampler.sample_asnc(tasks, args)`的整个流程

  首先我们需要将采样的任务`tasks`传送给`SamplerWorker`, 让他采集数据, 因为这个地方采集数据是异步并行的, 所以效率非常的高, 在他采集数据的同时, 我们进行数据的收集和整理, 让数据从下层传输到上层, 这个过程也就是`sonsumer`做的事情, 在将数据`put`进`task_queue`之后, 我们就运行`_start_consumer_threads`, 在这个里面我们`create`了会让程序阻塞的`train_future`和`valid_future`, 其他地方的`future`都是这里`future`的返回值, 这个`future`定义在这里我们就不需要管他了, 我们只需要知道他是丢进了当前这个`self._event_loop`就可以了, `create`了之后我们就需要定义一个`thread`, 这个`thread`的目的就是将`future`的值给设置好, 把`futures`的值设置好也是一开始我们的目的.
  
  这个`thread`他做的事情就是把我们采集到的数据传回去, 我们使用`SamplerWorker`对任务进行采样, 采样任务通过在`sampler_async`函数里`put`进入`task_queue`告知, `SamplerWorker`采样到的数据通过`put`进入`train_queue`告知, 告知了之后这个`thread`正在运行的`_create_consumer`就会把数据给拿到, 并且让`eventloop`带回把这个值设置给`future`, 为什么不直接在`_create_consumer`里设置`future`, 这是因为`future`的创建和设置值都必须在同一个线程里面.
  
  我们再来看`SamplerWorker`, 当我们把`task`给`task_queue`的时候, `SamplerWorker`就开始运行了, 因为在`run`里面有一个`self.task_queue.get()`, 所以只要把值给到了`task_queue`, 那么就会通知`run`函数执行下去而不需要继续等待了, 接着就会调用`sample`函数得到`valid`和`train`的`episode`, 他们都利用`create_episode`创建一个`episode`,  `create_episode`会调用`sample_trajectories`让这个环境`step`一步, 然后逐渐的走到结束为止, 这样一个`episode`就完成了.
  
  最后还在这个`sample`里面更新了`params`, 这个暂时不讨论, 后面在讨论`param`更新的时候说.
