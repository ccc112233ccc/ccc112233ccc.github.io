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

一项分布式训练任务横跨数千张加速卡，计算已经结束的机器必须等待最慢的那条通信链路。普通数据中心里不算严重的一次哈希碰撞，到了这里可能让整组昂贵的加速卡一起空转；一条突发流，又足以瞬间吃满 400G 网卡。

Meta 在 2024 年发表的这篇 RoCE 论文，记录了网络从单交换机原型扩展到多座数千卡集群的过程。它不是一个新算法的发布会，更像一份历时数年的工程病历：随机哈希不够均匀，静态绑路怕故障，中央流量工程又太复杂，经典拥塞控制到了 400G 甚至被直接关闭。

贯穿这些取舍的核心判断是：**AI 网络不能只把训练看成一堆普通流量。集体通信的结构、加速卡的摆放和网络路径必须被当作同一个系统设计。**

## 绕过主机，不等于绕过网络

一台训练服务器内部，八张加速卡可以通过专用高速互连直接通信。一旦任务跨出服务器，数据便要经过网卡和交换机；Meta 的 Grand Teton 节点为八张加速卡配置八张 400G 网卡，形成一对一映射。

远程直接内存访问的优势，是让数据直接在应用内存之间搬运，不必经过主机处理器反复复制。它的正式名称是 RDMA；RoCEv2 再把这类消息封装进以太网数据包，让应用保留熟悉的通信语义，底层则复用标准以太网生态。

论文 Figure 4 把这种节点结构画得很清楚：下层八张 GPU 通过 NVSwitch 互联，跨机数据则沿 PCIe 进入中间层的八张 400G RDMA NIC。图中 CPU 仍然存在，但不再是 GPU 数据面的必经之路。

![Grand Teton 节点内部由八张 GPU、NVSwitch、PCIe 交换芯片和八张 400G RDMA 网卡组成](/assets/img/posts/meta-roce/figure-04-grand-teton.png)

_论文 Figure 4：Grand Teton 训练节点。这里最值得看的是 GPU 与 400G RDMA NIC 的一对一配置；它解释了数据如何绕开 CPU，却不代表交换网络也被绕开。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

这解释了论文标题里的“RDMA over Ethernet”，却没有解释它为何难以扩展。真正的麻烦来自训练流量：它比普通数据中心流量更少、更整齐，也因此更容易撞在一起。

## 一张训练集群，两套物理网络

Meta 很早就把训练服务器同时接入两张独立网络。前端网络负责读取数据、保存 checkpoint 和记录日志；后端网络只承载 GPU 之间的训练通信。

![训练机架左侧连接前端网络，右侧的 GPU 网卡连接独立后端网络](/assets/img/posts/meta-roce/figure-05-front-back.png)

_论文 Figure 5：黄色区域是数据读取、checkpoint 和日志使用的前端网络，蓝色区域是 GPU 集体通信使用的后端网络，中间 AI Rack 同时接入两边。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

后端网络也经历了三代变化：最初是少量机架接一台中央交换机的星形 RoCEv1；随后变成由机架交换机和集群交换机构成的两级 Clos，Meta 称之为 AI Zone；当一个 Zone 容不下更大的模型时，又增加聚合层，把多座 Zone 连成数据中心规模。

跨 Zone 链路并没有盲目追求完全无阻塞，而是有意保留 oversubscription。训练调度器读取服务器在物理拓扑中的位置，用近似“最小割”的方式安排 rank，尽量让通信密集的 GPU 留在同一 Zone。网络容量不足的问题，部分被转化成了任务放置问题。

这一步已经显露出整篇论文的气质：硬件拓扑、路由和训练调度器之间不存在清晰的组织边界，只有一条共同的性能路径。

## AI 流量的三个坏习惯

普通数据中心里，海量短流会给 ECMP 提供丰富的随机性。训练 collective 恰好相反：通信对象固定、模式周期重复、单条流又极大。论文把它概括为三个特征。

Figure 3 的两张热力图给出了最直观的区别。左边的 ranking workload 接近 full mesh，绿色流量铺满矩阵；右边的 LLM workload 只留下少数稳定连接。白色区域越多，可供 ECMP 哈希打散的流就越少。

![Ranking 流量接近全互联，而 LLM 流量只集中在少数 GPU 对之间](/assets/img/posts/meta-roce/figure-03-traffic.png)

_论文 Figure 3：ranking 与 LLM 实验流量模式对比。它证明的是两类 workload 的连接熵不同，不是说所有 LLM 都严格服从右图。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

论文采集了 2023 年第四季度约 3 万个随机训练任务。AllReduce 与 AlltoAll(v) 主导数据并行模型，AllGather 与 ReduceScatter 则是全分片数据并行的核心；128 GPU 的生产统计里，一张 GPU 平均只有约 4 个 AllReduce queue pair，而 AlltoAll(v) 约为 15 个。

这份 trace 有一道必须保留的边界：统计明确排除了大于 128 GPU 的长尾，因此没有直接包含 LLM 任务。作者认为，多维并行会把上万 GPU 的大任务拆成数百 GPU 的 collective，所以 16–128 GPU 仍有代表性。这个理由合理，却不等于 3 万任务已经覆盖了大模型训练的全部网络行为。

## 四条象流，为什么会挤进同一条路

设想四条同样大小的 AllReduce 流和四条等价上行链路。传统 ECMP 对五元组做哈希，希望它们随机散开；理想情况是一条流占一条路，实际情况却可能有两条甚至三条流落在同一条路，旁边的链路仍然空闲。

训练任务每轮重复相同的 collective，流量又缺少变化，碰撞不会像互联网短流那样很快消失。一个随机选择变成了持续性拥塞，最慢的路径最终决定 collective 何时完成。

Meta 用最忙链路与平均链路的流数之比衡量不均衡。在 16 条链路、1000 条流的模拟中，普通 ECMP 的平均比值超过 1.2；生产流的熵更低，情况还会更差。解决这件事的过程，先后走过了四条路。

## 路由演化不是一条胜利直线

最初的 path pinning 把特定目的地固定到特定路径。机架被完整分配给同一任务、网络又没有故障时，它很有效；一旦任务只占半个机架，或者上行链路失效，流量便会重新挤在少数路径上，训练性能退化可超过 30%。

短期补救非常直接：把机架上行带宽加倍，用 2 倍网络容量掩盖碰撞。长期方案则回到 ECMP，但先在 collective library 中把一对网卡之间的消息分散到更多 queue pair，再让交换机把 RoCE 的目标 QP 也加入哈希。E-ECMP 配合 QP scaling，在 AllReduce benchmark 中相对普通 ECMP 最多提升 40%。

代价并没有消失。QP 是网卡里的稀缺资源，full-mesh 的 ranking workload 只配置 4 倍 scaling，层次化 collective 更多的 LLM workload 则使用 16 倍。网络因此开始依赖应用类型调参数，运维复杂度重新从交换机转移到了通信库。

## 中央控制更聪明，却没有统治所有集群

Meta 还部署了中央流量工程控制器。它把实时拓扑、链路状态、流量矩阵和训练任务位置收集起来，每隔约 30 秒运行约束最短路径算法，再把 `<源端口, 目标前缀>` 对应的下一跳写入交换机。

![中央流量工程系统从拓扑、训练任务和流量计数器收集状态，再向交换机下发路径](/assets/img/posts/meta-roce/figure-09-te.png)

_论文 Figure 9：Meta 的中央 TE 架构。阅读顺序可以从底部数据平面开始：Open/R 提供实时拓扑，FBOSS 提供流量计数；中间服务补上训练任务位置；顶部控制器计算路径并写回交换机。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

在 128 节点的受控测试中，E-ECMP 的链路利用率分布在 40%–90%，中央 TE 则把各链路稳定在约 80%；AllReduce 与 AlltoAll 的 normalized bus bandwidth 提高 5%–10%。这是一项真实收益，却远没有标题式的数量级变化。

更重要的结论来自部署之后。论文原先把多链路同时故障视为少见情况，生产中却比预期更常发生；中央 TE 在这些时候可能输给 E-ECMP，还带来控制软件、状态管理和计算负担。Meta 最终让 TE 主要服务 ranking AI Zone，在数据中心规模集群里仍选择 E-ECMP。

论文把 flowlet switching 视为下一步：交换机在流的间隙重新选择负载更低的路径，兼顾快速反应与较低控制复杂度。256–512 微秒的间隔在 200G 测试中把乱序降到每秒 1 个包以下，但作者写得很谨慎——它仍取决于早期部署信号，不是已经全面替换现网的结论。

## 最反直觉的一步：关掉 DCQCN

RoCE 常被描述成“无损以太网”。Meta 的实现使用基于优先级的流控 PFC 和单条 lossless queue，遇到罕见丢包则依靠 go-back-N 重传。PFC 能阻止缓存溢出，却可能把暂停逐跳向上游传播，制造 head-of-line blocking；DCQCN 的职责，原本就是在拥塞变成 PFC 之前让发送端降速。

但存储网络里的标准答案，并没有直接变成训练网络里的标准答案。在 200G AllReduce 测试中，更严格的 ECN 阈值让完成时间变差约 5%。作者继续扫描 DCQCN 参数，在 128 GPU AlltoAll 中只换来约 3% 的完成时间改善，PFC 指标却恶化 2–3 倍；论文还明确说，没有公开最极端 corner case 的数据。

![更严格的 ECN 阈值在不同 collective size 下没有缩短 AllReduce 完成时间](/assets/img/posts/meta-roce/figure-12-ecn.png)

_论文 Figure 12：32 张 GPU 的 AllReduce 测试。蓝柱采用更严格的 ECN 阈值，几乎没有更快，部分尺寸反而略慢。注意纵轴是 collective 完成时间，不是端到端训练时间。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

![多组放宽后的 DCQCN 参数与默认设置在 AlltoAll 完成时间上非常接近](/assets/img/posts/meta-roce/figure-13-dcqcn.png)

_论文 Figure 13：128 张 GPU 的 AlltoAll 参数扫描。五组柱子贴得很近，说明调参只带来几个百分点的变化；这张图没有展示随之恶化的 PFC 指标，也没有覆盖作者未公开的极端 corner case。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

到了 400G，默认配置配合翻倍的 ECN 阈值仍然退化，固件实现变化又带来 CNP 计数 bug 和可观测性下降。Meta 最终关闭了 400G 部署里的 DCQCN，并报告仅用 PFC 运行一年以上仍保持稳定。

这个决定很容易被压缩成一句错误的标题：“AI 网络不需要拥塞控制。”事实上，网络层算法退出之后，拥塞管理被上移到了 collective library。

## 真正接管入口的是接收端

Meta 生产环境里的 GPU 通信以接收端发起。发送端有数据，也不能立刻把 RDMA write 打进网络；它必须等接收端腾出 channel buffer，并返回 clear-to-send（CTS）消息。接收端处理越慢，CTS 发得越少，在途数据便自然收缩。

![发送端等待接收端 CTS 后才从 channel buffer 发起 RDMA write](/assets/img/posts/meta-roce/figure-14-receiver.png)

_论文 Figure 14：GPU 到 GPU 的接收端驱动通信。顺着中间箭头看，接收端先回 CTS，发送端才提交 RDMA write；接收侧完成复制并释放 channel buffer 后，下一轮数据才获得入口。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

Meta 进一步调整 channel 数量与 buffer 大小，并让交换机优先转发 CTS，避免控制消息被拥塞的数据流饿死。集群的 spine 使用按端口静态划分的深缓存，负责吸收瞬时突发，尽量不把反压扩散到其他端口。

论文用一次五分钟的 16:1 incast 对比说明效果：相同流量下，持续灌流的 Perftest 让反压一路传到发送网卡，而 NCCL Gather 在表中三个观察位置的 PFC 计数均为 0；把接收端带宽限制到 25% 后，接收端准入也能随之降速。

这是一个刻意选择的压力测试，Gather 在 Meta 生产训练中很少使用。它证明这套阀门能处理该实验里的严重 incast，却不能单独证明任意 collective、任意慢接收端和任意交换机缓存配置都可以安全关闭 DCQCN。

## 网络与 collective 必须一起调

现成 NCCL 的默认环境假设包括低于 10 微秒的 RTT、自适应路由和无 oversubscription 的拓扑。Meta 的生产网络使用 VOQ 交换机，空载 RTT 约为 22 微秒；控制消息、消息分块和 collective 拓扑的默认选择因此不再合适。

团队增大网络中的 message 与 channel buffer，调整 Tree、Ring 和低延迟协议的切换点，并提高 CTS 与 ACK 的优先级。等待 rendezvous 消息的 P90 延迟从 43 微秒降到 4 微秒；调整交换机 credit 后，延迟敏感的小消息 collective 最多再改善 15%。

![AlltoAll 和 AllReduce 在开箱配置、只调 collective、网络与 collective 协同调优下的带宽对比](/assets/img/posts/meta-roce/figure-15-cotuning.png)

_论文 Figure 15：上图是 AlltoAll，下图是 AllReduce。蓝柱代表开箱配置，绿柱只调整 collective，红柱同时调整网络与 collective；小消息区的差距尤其明显。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

作者报告许多 case 超过 2 倍提升。这里的关键词不是某个神奇参数，而是“协同”：GPU kernel、CPU proxy、通信库、网卡 credit、交换机队列和物理拓扑同时决定最终带宽。

## 生产系统的最后一层，是可观测性

Meta 用一个 200G AI Zone 的多年数据展示网络演化。第一阶段是 1:1 容量配比加静态路由，碰撞让性能低且分散；第二阶段把上行容量扩成 2 倍，性能上升，但硬件成本也翻倍；第三阶段引入 TE 后，带宽中位数相近，分布明显收紧；第四阶段把额外容量收回到约 1:1.125，仍为两条链路故障保留余量。

![四个网络阶段的 normalized bus bandwidth 分布由离散变得集中](/assets/img/posts/meta-roce/figure-16-stages.png)

_论文 Figure 16：同一座 AI Zone 多年演进后的 normalized AllReduce bus bandwidth。Stage 1 的分布最散；Stage 2 用 2 倍容量解决问题；Stage 3 用 TE 收紧波动；Stage 4 在保留故障余量的同时回收了大部分额外容量。来源：Gangidi et al., SIGCOMM ’24，[正式 PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)；仅裁切页面空白，图中数据与标注未修改。_

这段历史比单次 benchmark 更接近论文真正的贡献：新算法的价值，不只是峰值高多少，还包括能否回收昂贵容量、能否在故障时退回默认路由，以及运维团队是否承担得起复杂度。

论文列出的核心故障信号也很朴素：乱序包计数、链路 flap 和本地 ACK 超时。PFC 暂停超过 200 毫秒会触发 watchdog，机架交换机 buffer 超过 80% 则报警。RoCE 的性能问题往往跨越 GPU、PCIe、网卡、交换机和固件，单看某一层很容易得到“网络没有异常”的假象。

一个生产事故把这种耦合表现得尤其完整：旧代码错误假设了 spine 的 SRAM 容量，固件升级让网卡发得更猛，H100 又提高了 I/O 压力，三件事碰上突发训练流量才触发 tail drop。它不是某个经典拥塞算法可以单独修复的故障，而是端到端回归测试与基线监控的问题。

## 论文数字，各自证明了什么

| 结果 | 适用范围 | 不能推出的结论 |
| --- | --- | --- |
| Path pinning 退化超过 30% | 碎片化任务放置与故障场景 | 所有静态路由都会稳定退化 30% |
| E-ECMP + QP scaling 最多提升 40% | AllReduce benchmark，对比普通 ECMP | 所有模型端到端训练加速 40% |
| TE 比 E-ECMP 高 5%–10% | 128 节点受控 AllReduce / AlltoAll | TE 在故障和数据中心规模仍然最优 |
| 协同调优在许多 case 超过 2× | Figure 15 的 collective sizes 与网络环境 | 整个训练任务 wall-clock 缩短 2× |
| 3 万任务 trace | 2023Q4 随机任务，排除 >128 GPU 长尾 | 直接代表万卡 LLM 的全部流量分布 |
| 关闭 DCQCN 后稳定运行 | Meta 的 400G 拓扑、PFC、深缓存与接收端准入 | 任意 RoCE 网络都应关闭拥塞控制 |

这篇论文很少给出模型训练时间或成本的端到端对比，主要证据来自 collective benchmark、流级模拟、生产 normalized bandwidth 和运维事件。它证明 Meta 的 RoCE 路线可以在自身软硬件栈上扩展到数千 GPU 集群，也展示了真实部署如何迫使设计反复转向；它没有证明 RoCE 普遍优于 InfiniBand，也没有给出可直接复制的万能参数表。

## 一篇无法“复现”，但很值得保存的论文

这不是开源系统论文。论文与 Meta 官方文章没有提供实现仓库、原始生产 trace、TE 控制器代码或完整设备配置；截至 2026-08-17，使用论文标题、RoCET、E-ECMP 与 receiver-driven admission 检索，也没有找到对应公开代码。NCCL Tests 等底层工具是公开的，但不能替代 Meta 的生产系统。

不可复现并不等于没有价值，只是价值类型不同。它最可信的部分是多年部署过程中留下的失败路径：两倍带宽是一剂昂贵止痛药，中央控制会输给故障频率，经典拥塞控制可能不如上层准入，网络性能最终取决于通信库与物理 fabric 的共同假设。

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
| 代码状态 | 未发现论文系统对应的公开实现或原始数据；核验日期 2026-08-17 |

---

文中论文图来自 Gangidi et al., SIGCOMM ’24 原文，均按解读需要裁去页面空白，未修改图内数据、坐标或图例；每张图的编号与正式 PDF 链接见对应图注。题图由 burger / Parallel Bites 制作，正文数字与实验边界依据论文原文核验。
