# maml-rl

+ `futures = sampler.sample_asnc(tasks, args)`的整个流程

  首先我们需要将采样的任务`tasks`传送给`SamplerWorker`, 让他采集数据, 因为这个地方采集数据是异步并行的, 所以效率非常的高, 在他采集数据的同时, 我们进行数据的收集和整理, 让数据从下层传输到上层, 这个过程也就是`sonsumer`做的事情, 在将数据`put`进`task_queue`之后, 我们就运行`_start_consumer_threads`, 在这个里面我们`create`了会让程序阻塞的`train_future`和`valid_future`, 其他地方的`future`都是这里`future`的返回值, 这个`future`定义在这里我们就不需要管他了, 我们只需要知道他是丢进了当前这个`self._event_loop`就可以了, `create`了之后我们就需要
