---
title: "当一条流就能吃满 400G：Meta 如何把以太网变成 AI 训练网络"
date: 2026-08-17 18:00:00 +0800
categories: [AI Infrastructure, Networking]
tags: [Meta, RoCE, RDMA, Distributed Training, Traffic Engineering, Congestion Control]
description: "Meta 没有靠一个万能算法连接数千张 GPU，而是重做了后端拓扑、路由、流量准入和运维。最反直觉的一步，是在 400G 集群里关掉 DCQCN。"
image:
  path: /assets/img/posts/meta-roce/hero-meta-roce.svg
  alt: GPU nodes connected through a dedicated RoCE Ethernet fabric with a highlighted routing collision.
toc: true
pin: false
---

一项分布式训练任务横跨数千张加速卡，计算已经结束的机器必须等待最慢的那条通信链路。本应走不同路径的几股流量偏偏被分到同一条链路——这就是哈希碰撞；在普通数据中心里它未必严重，到了训练集群却可能让整组昂贵的加速卡一起空转。一条突发流，又足以瞬间吃满 400G 网卡。

Meta 在 2024 年发表的这篇 RoCE 论文，记录了网络从单交换机原型扩展到多座数千卡集群的过程。它不是一个新算法的发布会，更像一份历时数年的工程病历：随机哈希不够均匀，静态绑路怕故障，中央流量工程又太复杂，经典拥塞控制到了 400G 甚至被直接关闭。

贯穿这些取舍的核心判断是：**AI 网络不能只把训练看成一堆普通流量。[集体通信](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)（collective，一组 GPU 共同完成的一次数据交换）的结构、加速卡的摆放和网络路径必须被当作同一个系统设计。**

## 绕过主机，不等于绕过网络

一台训练服务器内部，八张 GPU（这里就是负责训练计算的加速卡）可以通过专用高速互连直接通信。一旦任务跨出服务器，数据便要经过网卡和交换机；Meta 的 Grand Teton 节点为八张 GPU 配置八张 400G 网卡，形成一对一映射。

远程直接内存访问（[RDMA](https://docs.nvidia.com/networking/display/mlnxofedv23101190lts/rdma%2Bover%2Bconverged%2Bethernet%2B%28roce%29)）让网卡直接在两台机器的应用内存之间搬数据，不必让主机 CPU 逐份复制。RoCEv2 的全称是 RDMA over Converged Ethernet v2，它把 RDMA 消息装进可路由的 UDP/IP 以太网包：上层仍像在做远程内存读写，底层可以穿过普通三层以太网。

论文 Figure 4 把这种节点结构画得很清楚：下层八张 GPU 通过 NVSwitch（机内 GPU 高速交换芯片）互联；跨机数据则沿 PCIe（服务器内部的高速总线）进入中间层的八张 400G RDMA NIC。NIC 的全称是 Network Interface Card，也就是网卡。图中 CPU 仍然存在，但不再是 GPU 数据面的必经之路。

![Grand Teton 节点内部由八张 GPU、NVSwitch、PCIe 交换芯片和八张 400G RDMA 网卡组成](/assets/img/posts/meta-roce/figure-04-grand-teton.png)

_论文 Figure 4：Grand Teton 训练节点。这里最值得看的是 GPU 与 400G RDMA NIC 的一对一配置；它解释了数据如何绕开 CPU，却不代表交换网络也被绕开。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

这解释了论文标题里的“RDMA over Ethernet”，却没有解释它为何难以扩展。真正的麻烦来自训练流量：它比普通数据中心流量更少、更整齐，也因此更容易撞在一起。

## 一张训练集群，两套物理网络

Meta 很早就把训练服务器同时接入两张独立网络。前端网络负责读取数据、保存 checkpoint（训练中途保存的模型快照）和记录日志；后端网络只承载 GPU 之间的训练通信。

![训练机架左侧连接前端网络，右侧的 GPU 网卡连接独立后端网络](/assets/img/posts/meta-roce/figure-05-front-back.png)

_论文 Figure 5：黄色区域是数据读取、checkpoint 和日志使用的前端网络，蓝色区域是 GPU 集体通信使用的后端网络，中间 AI Rack 同时接入两边。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

后端网络也经历了三代变化：最初是少量机架接一台中央交换机的星形 RoCEv1；随后变成两级 Clos——每台机架交换机都可以通过多台上层交换机到达别的机架，路径不止一条。Meta 把这样一组相对独立的训练网络称为 AI Zone；当一个 Zone 容不下更大的模型时，又增加聚合层，把多座 Zone 连成数据中心规模。

跨 Zone 链路并没有盲目追求“所有机器同时满速仍不堵”，而是有意保留 oversubscription（过订阅，例如机架内部合计 8 份带宽，却只配 4 份上行带宽）。训练调度器读取服务器的物理位置，再安排每张 GPU 的 rank——也就是它在训练任务里的编号和角色——尽量让通信最频繁的 GPU 留在同一 Zone。网络容量不足的问题，部分被转化成了任务放置问题。

这一步已经显露出整篇论文的气质：硬件拓扑、路由和训练调度器之间不存在清晰的组织边界，只有一条共同的性能路径。

## AI 流量的三个坏习惯

普通数据中心里，海量短流会给 [ECMP](https://www.rfc-editor.org/info/rfc2992/) 提供丰富的随机性。ECMP 的全称是“等价多路径”：当交换机发现几条同样短的路时，它用流的包头信息做哈希，为这条流固定选其中一路。训练 collective 恰好相反：通信对象固定、模式周期重复、单条流又极大。论文把它概括为三个特征。

Figure 3 的两张热力图给出了最直观的区别。左边的推荐排序任务接近 full mesh（每台机器都和很多机器通信），绿色流量铺满矩阵；右边的大模型任务只留下少数稳定连接。白色区域越多，可供 ECMP 哈希打散的流就越少。

![Ranking 流量接近全互联，而 LLM 流量只集中在少数 GPU 对之间](/assets/img/posts/meta-roce/figure-03-traffic.png)

_论文 Figure 3：推荐排序任务与大模型任务的流量模式对比。它证明的是两类任务可供哈希区分的连接组合数量不同，不是说所有大模型都严格服从右图。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

先把几个 collective 名字翻译成动作。AllReduce 是“每张 GPU 交出一份数字，求和或求平均后，每张 GPU 都拿到同一份结果”；例如四张卡分别有 `1、2、3、4`，求和后四张卡都得到 `10`。AlltoAll 是每张卡分别给其他卡发送不同的数据块，名称后的 `v` 表示各块大小可以不同。AllGather 是大家各交一块、最后每张卡都拿到全集；ReduceScatter 则是先汇总，再把结果切开分给大家。

论文采集了 2023 年第四季度约 3 万个随机训练任务。AllReduce 与 AlltoAll(v) 主导数据并行模型，AllGather 与 ReduceScatter 则是全分片数据并行的核心。在 128 GPU 的生产统计里，一张 GPU 平均只有约 4 个 AllReduce Queue Pair（QP，队列对）；它可以理解成网卡里一条 RDMA 逻辑连接所使用的一对收发队列。AlltoAll(v) 平均约有 15 个 QP，依然谈不上“海量随机流”。

这份 trace（线上流量记录）有一道必须保留的边界：统计明确排除了大于 128 GPU 的长尾，因此没有直接包含 LLM 任务。作者认为，多维并行会把上万 GPU 的大任务拆成数百 GPU 的 collective，所以 16–128 GPU 仍有代表性。这个理由合理，却不等于 3 万任务已经覆盖了大模型训练的全部网络行为。

## 四条象流，为什么会挤进同一条路

设想四条同样大小的 AllReduce 流和四条等价上行链路，编号为 0、1、2、3。交换机不会先观察哪条路空闲，再做全局排座；它只读取每条流的“五元组”——源 IP、目标 IP、源端口、目标端口和传输协议——算出一个哈希数，再用这个数选择路径。

假设四条流算出的哈希数恰好是 6、10、11、15，路径规则是“哈希数除以 4，取余数”：

| 流 | 哈希数 | `mod 4` 后选择 |
| --- | ---: | ---: |
| A | 6 | 链路 2 |
| B | 10 | 链路 2 |
| C | 11 | 链路 3 |
| D | 15 | 链路 3 |

结果是链路 2 和 3 各挤了两条流，链路 0 和 1 却完全空闲。哈希保证的是“同一条流稳定地走同一路”，从而减少乱序；它并不知道各条链路此刻已经坐了几条流，因此“总体随机”并不等于“这四条一定一人一路”。即使假设每条流都独立、均匀地四选一，四条流刚好一人一路的概率也只有 `4! / 4⁴ = 9.375%`，其余情况都会发生至少一次碰撞。

训练任务每轮重复相同的 collective，五元组和 QP 往往也不变；哈希是确定性的，同一组输入下一轮仍会选中同一路。碰撞不会像互联网短流那样很快消失，一个随机选择便变成了持续性拥塞，最慢的路径最终决定 collective 何时完成。

Meta 用最忙链路与平均链路的流数之比衡量不均衡。在 16 条链路、1000 条流的模拟中，普通 ECMP 的平均比值超过 1.2；生产流可供哈希区分的字段变化更少，情况还会更差。解决这件事的过程，先后走过了四条路。

## 路由演化不是一条胜利直线

最初的 path pinning（路径绑定）像提前给每个目的地分配固定车道。机架被完整分配给同一任务、网络又没有故障时，它很有效；一旦任务只占半个机架，或者上行链路失效，原先精心对齐的“座位表”便被打乱，流量会重新挤在少数路径上，训练性能退化可超过 30%。

短期补救非常直接：把机架上行带宽加倍，用 2 倍网络容量掩盖碰撞。长期方案则回到 ECMP，但先让 collective library（集体通信库）在同一对网卡之间创建更多 QP，相当于把一股大车流拆成更多可单独分路的车队；再让交换机把 RoCE 的目标 QP 编号也加入哈希。这个定制版称为 E-ECMP，增加 QP 数量称为 QP scaling。两者配合，在 AllReduce 基准测试中相对普通 ECMP 最多提升 40%。

代价并没有消失。QP 是网卡里的稀缺资源，近似全互联的推荐排序任务只配置 4 倍 scaling，层次化 collective 更多的大模型任务则使用 16 倍。网络因此开始依赖应用类型调参数，运维复杂度重新从交换机转移到了通信库。

## 中央控制更聪明，却没有统治所有集群

Meta 还部署了中央流量工程（Traffic Engineering，TE）控制器。ECMP 像每辆车自己抽签，TE 则像一个看到全城路况的导航中心：它收集实时拓扑、链路状态、各处流量和训练任务位置，每隔约 30 秒重新计算路径，再把 `<源端口, 目标网段>` 应走的下一跳写入交换机。

![中央流量工程系统从拓扑、训练任务和流量计数器收集状态，再向交换机下发路径](/assets/img/posts/meta-roce/figure-09-te.png)

_论文 Figure 9：Meta 的中央 TE 架构。阅读顺序可以从底部数据平面开始：Open/R（路由软件）提供实时拓扑，FBOSS（交换机控制软件）提供流量计数；中间服务补上训练任务位置；顶部控制器计算路径并写回交换机。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

在 128 节点的受控测试中，E-ECMP 的链路利用率分布在 40%–90%，中央 TE 则把各链路稳定在约 80%；AllReduce 与 AlltoAll 的 normalized bus bandwidth 提高 5%–10%。这个指标把 collective 实际搬运数据的能力折算成相对可用网络能力的百分比，便于比较不同配置。这是一项真实收益，却远没有标题式的数量级变化。

更重要的结论来自部署之后。论文原先把多链路同时故障视为少见情况，生产中却比预期更常发生；中央 TE 在这些时候可能输给 E-ECMP，还带来控制软件、状态管理和计算负担。Meta 最终让 TE 主要服务 ranking AI Zone，在数据中心规模集群里仍选择 E-ECMP。

论文把 flowlet switching（流片交换）视为下一步。一条长流往往不是持续不断地发包，中间会出现短暂空隙；交换机把空隙前后的两段看成不同 flowlet，只在空隙处为下一段改选更空闲的路径。这样比等待 30 秒的中央 TE 反应更快，又尽量避免在连续发包中途换路造成乱序。256–512 微秒的间隔在 200G 测试中把乱序降到每秒 1 个包以下，但作者写得很谨慎——它仍取决于早期部署信号，不是已经全面替换现网的结论。

## 最反直觉的一步：关掉 DCQCN

把交换机缓存想成一个蓄水池。水位快满时，[PFC](https://docs.nvidia.com/networking/display/OFEDv490170/Flow%2BControl)（Priority Flow Control，基于优先级的流控）会向上一跳发送“这类流量先暂停”的信号，防止缓存溢出。Meta 把 RoCE 放在一条 lossless queue（目标是尽量不丢包的队列）里；真遇到罕见丢包，则采用 go-back-N，即从缺失的数据包开始把后面一段全部重传。

PFC 更像水位已经危险时拉下的紧急刹车。暂停信号可能逐跳向发送端蔓延，还可能让队首被堵住的数据包挡住后面本可通行的数据，这叫 head-of-line blocking（队头阻塞）。DCQCN 的全称是 Data Center Quantized Congestion Notification，是一套更早介入的 RoCE 拥塞控制：交换机在水位达到 ECN（Explicit Congestion Notification，显式拥塞通知）阈值时给数据包做标记，接收端回传拥塞通知，发送端据此降速，尽量别等到 PFC 真正触发。

但存储网络里的标准答案，并没有直接变成训练网络里的标准答案。ECN 阈值越低，交换机越早发出“开始拥塞”的信号，也就越容易让发送端提前减速。在 200G AllReduce 测试中，更严格的 ECN 阈值反而让完成时间变差约 5%。作者继续扫描 DCQCN 参数，在 128 GPU AlltoAll 中只换来约 3% 的完成时间改善，PFC 指标却恶化 2–3 倍；论文还明确说，没有公开最极端边界场景的数据。

![更严格的 ECN 阈值在不同 collective size 下没有缩短 AllReduce 完成时间](/assets/img/posts/meta-roce/figure-12-ecn.png)

_论文 Figure 12：32 张 GPU 的 AllReduce 测试。蓝柱采用更严格的 ECN 阈值，几乎没有更快，部分尺寸反而略慢。注意纵轴是 collective 完成时间，不是端到端训练时间。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

![多组放宽后的 DCQCN 参数与默认设置在 AlltoAll 完成时间上非常接近](/assets/img/posts/meta-roce/figure-13-dcqcn.png)

_论文 Figure 13：128 张 GPU 的 AlltoAll 参数扫描。五组柱子贴得很近，说明调参只带来几个百分点的变化；这张图没有展示随之恶化的 PFC 指标，也没有覆盖作者未公开的极端边界场景。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

到了 400G，默认配置配合翻倍的 ECN 阈值仍然退化，固件实现变化又带来 CNP（Congestion Notification Packet，拥塞通知包）计数 bug 和可观测性下降。Meta 最终关闭了 400G 部署里的 DCQCN，并报告仅用 PFC 运行一年以上仍保持稳定。

这个决定很容易被压缩成一句错误的标题：“AI 网络不需要拥塞控制。”事实上，网络层算法退出之后，拥塞管理被上移到了 collective library。

## 真正接管入口的是接收端

Meta 生产环境里的 GPU 通信以接收端发起。发送端有数据，也不能立刻执行 RDMA write（让网卡直接写入远端内存）；它必须等接收端腾出 channel buffer（通信过程中的中转缓冲区），并返回 clear-to-send（CTS，可以发送）消息。接收端处理越慢，CTS 发得越少，在途数据便自然收缩。

![发送端等待接收端 CTS 后才从 channel buffer 发起 RDMA write](/assets/img/posts/meta-roce/figure-14-receiver.png)

_论文 Figure 14：GPU 到 GPU 的接收端驱动通信。顺着中间箭头看，接收端先回 CTS，发送端才提交 RDMA write；接收侧完成复制并释放 channel buffer 后，下一轮数据才获得入口。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

Meta 进一步调整 channel 数量与 buffer 大小，并让交换机优先转发 CTS，避免控制消息被拥塞的数据流饿死。集群的 spine（连接多台机架交换机的上层核心交换机）使用按端口静态划分的深缓存，负责吸收瞬时突发，尽量不把反压扩散到其他端口。

论文用一次五分钟的 16:1 incast（16 个发送端同时向 1 个接收端灌数据）对比说明效果。相同流量下，持续灌流的 Perftest 压测工具让反压一路传到发送网卡；NCCL Gather——把多张卡的数据集中到一张卡——在表中三个观察位置的 PFC 计数均为 0。把接收端带宽限制到 25% 后，接收端准入也能随之降速。

这是一个刻意选择的压力测试，Gather 在 Meta 生产训练中很少使用。它证明这套阀门能处理该实验里的严重 incast，却不能单独证明任意 collective、任意慢接收端和任意交换机缓存配置都可以安全关闭 DCQCN。

## 网络与 collective 必须一起调

NCCL 是 NVIDIA 提供的 GPU 集体通信库。它的默认环境假设包括低于 10 微秒的 RTT（Round-Trip Time，数据往返一次所需时间）、自适应路由和无过订阅拓扑。Meta 的生产网络使用 VOQ（Virtual Output Queue，虚拟输出队列）交换机；VOQ 会按目标出口分别排队，避免一个拥堵出口挡住其他出口。这里空载 RTT 约为 22 微秒，控制消息、消息分块和 collective 拓扑的默认选择因此不再合适。

团队增大网络中的 message 与 channel buffer，并调整 Tree 与 Ring 的切换点：前者像树一样分层汇总，后者让 GPU 沿环逐站传递数据。CTS 和 ACK（Acknowledgement，确认消息）获得更高优先级，避免控制消息排在大数据包后面。

发送端等待 rendezvous（通信双方确认“内存和接收方都准备好了”）消息的 P90 延迟从 43 微秒降到 4 微秒。P90 的意思是 90% 的等待都不超过这个数，它比只看平均值更能暴露慢请求。调整交换机 credit——可以理解为下游提前发放的“还能接收多少数据”的额度——之后，延迟敏感的小消息 collective 最多再改善 15%。

![AlltoAll 和 AllReduce 在开箱配置、只调 collective、网络与 collective 协同调优下的带宽对比](/assets/img/posts/meta-roce/figure-15-cotuning.png)

_论文 Figure 15：上图是 AlltoAll，下图是 AllReduce。蓝柱代表开箱配置，绿柱只调整 collective，红柱同时调整网络与 collective；小消息区的差距尤其明显。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

作者报告许多测试场景超过 2 倍提升。这里的关键词不是某个神奇参数，而是“协同”：GPU 计算程序、负责通信的 CPU 线程、通信库、网卡额度、交换机队列和物理拓扑同时决定最终带宽。

## 生产系统的最后一层，是可观测性

Meta 用一个 200G AI Zone 的多年数据展示网络演化。第一阶段是 1:1 容量配比加静态路由，碰撞让性能低且分散；第二阶段把上行容量扩成 2 倍，性能上升，但硬件成本也翻倍；第三阶段引入 TE 后，带宽中位数相近，分布明显收紧；第四阶段把额外容量收回到约 1:1.125，仍为两条链路故障保留余量。

![四个网络阶段的 normalized bus bandwidth 分布由离散变得集中](/assets/img/posts/meta-roce/figure-16-stages.png)

_论文 Figure 16：同一座 AI Zone 多年演进后的 normalized AllReduce bus bandwidth。Stage 1 的分布最散；Stage 2 用 2 倍容量解决问题；Stage 3 用 TE 收紧波动；Stage 4 在保留故障余量的同时回收了大部分额外容量。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

这段历史比单次基准测试更接近论文真正的贡献：新算法的价值，不只是峰值高多少，还包括能否回收昂贵容量、能否在故障时退回默认路由，以及运维团队是否承担得起复杂度。

论文列出的核心故障信号也很朴素：乱序包计数、链路 flap（端口在“可用/不可用”之间反复抖动）和本地 ACK 超时。PFC 暂停超过 200 毫秒会触发 watchdog（检测到长时间卡死后主动干预的看门狗），机架交换机 buffer 超过 80% 则报警。RoCE 的性能问题往往跨越 GPU、PCIe、网卡、交换机和固件，单看某一层很容易得到“网络没有异常”的假象。

一个生产事故把这种耦合表现得尤其完整：旧代码错误假设了 spine 的 SRAM（交换芯片内部用于缓存的高速存储）容量，固件升级让网卡发得更猛，H100 又提高了 I/O 压力，三件事碰上突发训练流量才触发 tail drop——缓存装满后，新到的数据包直接被丢弃。它不是某个经典拥塞算法可以单独修复的故障，而是端到端回归测试与基线监控的问题。

## 论文数字，各自证明了什么

| 结果 | 适用范围 | 不能推出的结论 |
| --- | --- | --- |
| Path pinning 退化超过 30% | 碎片化任务放置与故障场景 | 所有静态路由都会稳定退化 30% |
| E-ECMP + QP scaling 最多提升 40% | AllReduce 基准测试，对比普通 ECMP | 所有模型端到端训练加速 40% |
| TE 比 E-ECMP 高 5%–10% | 128 节点受控 AllReduce / AlltoAll | TE 在故障和数据中心规模仍然最优 |
| 协同调优在许多场景超过 2× | Figure 15 的消息大小与网络环境 | 整个训练任务实际耗时缩短 2× |
| 3 万任务 trace | 2023Q4 随机任务，排除 >128 GPU 长尾 | 直接代表万卡 LLM 的全部流量分布 |
| 关闭 DCQCN 后稳定运行 | Meta 的 400G 拓扑、PFC、深缓存与接收端准入 | 任意 RoCE 网络都应关闭拥塞控制 |

这篇论文很少给出模型训练时间或成本的端到端对比，主要证据来自 collective 基准测试、流级模拟、生产环境的归一化带宽和运维事件。它证明 Meta 的 RoCE 路线可以在自身软硬件栈上扩展到数千 GPU 集群，也展示了真实部署如何迫使设计反复转向；它没有证明 RoCE 普遍优于 InfiniBand，也没有给出可直接复制的万能参数表。

## 一篇无法“复现”，但很值得保存的论文

这不是开源系统论文。论文与 Meta 官方文章没有提供实现仓库、原始生产 trace、TE 控制器代码或完整设备配置；截至 2026-08-18，使用论文标题、RoCET、E-ECMP 与 receiver-driven admission 检索，也没有找到对应公开代码。NCCL Tests 等底层工具是公开的，但不能替代 Meta 的生产系统。

不可复现并不等于没有价值，只是价值类型不同。它最可信的部分是多年部署过程中留下的失败路径：两倍带宽是一剂昂贵止痛药，中央控制会输给故障频率，经典拥塞控制可能不如上层准入，网络性能最终取决于通信库与整张物理交换网络的共同假设。

最值得带走的判断也不是“RoCE 足以替代专用互连”。这篇论文真正展示的是：**AI 集群网络的优化单位已经不再是一条流、一台交换机或一个拥塞算法，而是从模型 collective 一直延伸到网卡与拓扑的整条通信链。** 以太网之所以能被推到这个规模，不是因为它天然适合训练，而是因为周围每一层都被重新安排过。

## 论文与核验资料

| 项目 | 内容 |
| --- | --- |
| 论文 | *RDMA over Ethernet for Distributed AI Training at Meta Scale* |
| 作者 | Adithya Gangidi, Rui Miao, Shengbao Zheng 等，Meta |
| 发表 | ACM SIGCOMM 2024，14 pages |
| DOI | [10.1145/3651890.3672233](https://doi.org/10.1145/3651890.3672233) |
| 正式 PDF | [Meta Engineering PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf) |
| 官方介绍 | [RoCE networks for distributed AI training at scale](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/) |
| 入门资料 | [NVIDIA：Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) · [NVIDIA：RoCE 基础](https://docs.nvidia.com/networking/display/mlnxofedv23101190lts/rdma%2Bover%2Bconverged%2Bethernet%2B%28roce%29) · [RFC 2992：ECMP 哈希算法](https://www.rfc-editor.org/info/rfc2992/) |
| 代码状态 | 未发现论文系统对应的公开实现或原始数据；核验日期 2026-08-18 |

---

文中论文图来自 Gangidi et al., SIGCOMM ’24 原文，均按解读需要裁去页面空白，未修改图内数据、坐标或图例；每张图的编号与正式 PDF 链接见对应图注。题图由 burger / Parallel Bites 制作，正文数字与实验边界依据论文原文核验。
