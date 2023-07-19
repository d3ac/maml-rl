# maml-rl

+ `futures = sampler.sample_asnc(tasks, args)`的整个流程

  首先我们需要将采样的任务`tasks`传送给`SamplerWorker`, 让他采集数据, 因为这个地方采集数据是异步并行的, 所以效率非常的高, 在他采集数据的同时, 我们进行数据的收集和整理, 让数据从下层传输到上层, 这个过程也就是`sonsumer`做的事情, 在将数据`put`进`task_queue`之后, 我们就运行`_start_consumer_threads`, 在这个里面我们`create`了会让程序阻塞的`train_future`和`valid_future`, 其他地方的`future`都是这里`future`的返回值, 这个`future`定义在这里我们就不需要管他了, 我们只需要知道他是丢进了当前这个`self._event_loop`就可以了, `create`了之后我们就需要定义一个`thread`, 这个`thread`的目的就是将`future`的值给设置好, 把`futures`的值设置好也是一开始我们的目的.
  
  这个`thread`他做的事情就是把我们采集到的数据传回去, 我们使用`SamplerWorker`对任务进行采样, 采样任务通过在`sampler_async`函数里`put`进入`task_queue`告知, `SamplerWorker`采样到的数据通过`put`进入`train_queue`告知, 告知了之后这个`thread`正在运行的`_create_consumer`就会把数据给拿到, 并且让`eventloop`带回把这个值设置给`future`, 为什么不直接在`_create_consumer`里设置`future`, 这是因为`future`的创建和设置值都必须在同一个线程里面.
  
  我们再来看`SamplerWorker`, 当我们把`task`给`task_queue`的时候, `SamplerWorker`就开始运行了, 因为在`run`里面有一个`self.task_queue.get()`, 所以只要把值给到了`task_queue`, 那么就会通知`run`函数执行下去而不需要继续等待了, 接着就会调用`sample`函数得到`valid`和`train`的`episode`, 他们都利用`create_episode`创建一个`episode`,  `create_episode`会调用`sample_trajectories`让这个环境`step`一步, 然后逐渐的走到结束为止, 这样一个`episode`就完成了.
  
  最后还在这个`sample`里面更新了`params`, 这个暂时不讨论, 后面在讨论`param`更新的时候说.

+ TRPO

  TRPO的目标函数是 :

  $$
  L(\theta) = \sum_t \rho_t(\theta) A_t
  $$

  其中$\rho_t(\theta)$是新旧策略的概率之差, $A_t$是优势函数, 在优化目标函数的同时需要满足KL散度约束 : 

  $$
  KL[\pi(\theta_{\text{old}}) \| \pi(\theta)] \leq \delta
  $$

  拉格朗日函数是一种将优化问题的目标函数和约束条件组合起来的方法, 使用了一个拉格朗日乘子$\lambda$来将约束条件转化成一个可以优化的形式, 然后将目标函数和约束条件组合成一个求和的形式, 并且调整$\lambda$的大小来控制目标函数和约束条件之间的权衡, 具体的来说我们希望最大化$L(\theta)$, 最小化$KL$, 也就是最大化$-KL$, 我们可以调整这个$\lambda$, 当$\lambda$越大的时候, 违反$KL$约束的代价也会越大, 当$\lambda$越小的时候, 满足目标函数就会越重要.
  
  $$
  L(\theta) - \lambda KL[\pi(\theta_{\text{old}}) \| \pi(\theta)]
  $$

  为了让式子更便于优化, 我们先把$\theta$换成$\theta_{old}+\Delta\theta$, 这个时候我们就把优化的参数从$\theta$转换成了$\Delta\theta$了, 然后再进行二阶泰勒展开 : 

  $$
  L(\theta) \approx L(\theta_{old}) + \Delta \theta^T \nabla         L(\theta_{old}) + \frac{1}{2}\Delta \theta^T \nabla^2 L(\theta_{old})     \Delta \theta \\
  KL[\pi(\theta_{old}) \| \pi(\theta)] \approx \frac{1}{2}\Delta   \theta^T H \Delta \theta
  $$

  在这里再解释一下为什么$KL$项等于这个, 因为$KL(\theta)=KL[\pi(\theta_{old}) \|    \pi(\theta)]$, 所以他的二阶泰勒逼近是 : 

  $$
  KL(\theta) \approx KL(\theta_0) + KL'(\theta_0)(\theta - \theta_0) +   \frac{1}{2}KL''(\theta_0)(\theta - \theta_0)^2 \\
  
  \nabla_{\theta} D_{KL}(\pi(\theta_{\text{old}}) || \pi(\theta)) = E_{x \sim \pi(\theta_{\text{old}})}\left[\frac{\pi(\theta_{\text{old}}) - \pi(\theta)}{\pi(\theta)}\right]\\
  $$
   因为这里是一个多元函数, 所以泰勒展开会稍微变一下, 举一个二维函数变量为$x$和$y$的例子:

  $$
  f(x, y) \approx f(x_0, y_0) + f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0) + \frac{1}{2}(x - x_0)^2 f_{xx}(x_0, y_0) + (x - x_0)(y - y_0)f_{xy}(x_0, y_0) + \frac{1}{2}(y - y_0)^2 f_{yy}(x_0, y_0)
  $$

  后面的三项也就是二阶泰勒展开, 可以变一下成下面这样子:

  $$
  \frac{1}{2}(x - x_0, y - y_0) \begin{pmatrix}f_{xx}(x_0, y_0) & f_{xy}(x_0, y_0) \\ f_{yx}(x_0, y_0) & f_{yy}(x_0, y_0) \end{pmatrix} \begin{pmatrix}x - x_0 \\ y - y_0 \end{pmatrix}
  $$

  然后因为我们最后要关心的是两个策略之间的$\theta$不能超过某一个值, 所以这个时候我们就可以不要常数项第一项,因为最后会减去就没有了, 然后这里的步长是这样计算的 :

  $$
  \lambda = \sqrt{\frac{1}{2}\Delta\theta^T H \Delta\theta / \delta}\\
  \text{stepsize} = \frac{1}{\lambda}
  $$
