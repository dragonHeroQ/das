## 研发备忘录

* [Y] 回归DeepArchiRegressor的一套
* [0118] ArchiBlockClassifier参数空间用一套
* [0120] ArchiLayerClassifier get_params, set_params行为
* [0121] 时间budget
* [0121] 顺序的cross_validate
* [Y] 除RS外的架构搜索方法，BO, UCB, HB3
* [ ] 分布式fit
* [Y] 降低fit失败率， 如 LGBx4+MLPx1
* [ ] Parameter Sharing 优化
* [Y] reward记录时间，错误信息
* [Y] 修缮fit_predict
* [ ] HyperBand方法
* [Y] HyperBandIteration跑通（0121备忘）
* [Y] 实现DeepArchiClassifier的诸多方法
* [Y] 统一实验脚本系统
* [-] 层数没长完，记录及时结果，不要一棒子打死
* [-] 测试集性能曲线
* [-] 得到最好模型和参数后，能够重跑出性能
* [Y] 完善plot_learning_curve(s)
* [ ] 根据CPU和Memory情况，智能决定并发度
* [ ] CPU滑动窗口，Memory滑动窗口
* [Y] 修改score_to_loss, loss_to_score规则，照顾accuracy_score, r2_score
* [-] SMBO，随机算法是选取架构，要不要考虑架构中的参数？
* [Y] Sampling Strategy
* [Y] 考虑Confidence Screen方法来降低深度架构的耗时
* [Y] 根据Confidence Screen论文，考虑将early_stopping_rounds设置为1
* [Y] 结构搜索与超参优化协同(e.g. BO+UCB) => AEHT
* [Y] 不允许BO，RS采样重复的结构去运行，可以令Evaluator记录评估历史（自己记录）
* [Y] BO+UCB中，存储BO产生的最好的n个learningTools，以便UCB可以尝试多种nbandit+nstage的组合
* [Y] Evolutionary Algorithm for Architecture Search实现
* [Y] 分布式在全搜索空间进行超过10h的搜索
* [Y] 优生优育计划。在有的10h搜索中，竟然出现了48个超时，每个都跑了480s，
        也就是说，23040s都花在了无用搜索上，真正的有效搜索只有12960/36000=36%的时间。
        虽然超时的往往是SVC，GPC，但是也不能完全打死，比如SVC在adult数据集上绝对超时，但是在yeast上可以运行出来。
        所以，我们不仅应该尽力去搜索那些好的参数，也要尽力去避免明显无效的搜索。
* [-] 交叉验证输入缓存计划。每次深度结构的第一层时，交叉验证的输入是一致的，可以put到plasma store里面，以后可以复用，而不是每个都put一次。
注：很难做到，即使第一层缓存了，后面的层一直在变而且可能很大，最终仍会很快爆满。并且开进程训练仍会涉及到一次输入拷贝。
解决方案：按照ForestLayer的并行化方式，对每个unit单独地做交叉验证，最后给出结果就行。
* [ ] 调参计划。结构搜索进行到后面，可以逐渐演变为只对超参进行mutate，即只进行超参调优。分为两个阶段？结构选择+超参调优？
* [ ] 进行结构搜索中的超参调优时，考虑加上BayesianOptimization来采样参数。
* [Y] 分布式运行时，交叉验证交给各个unit自己来做。
* [ ] 增设distribute=3，由ArchiLayer亲自摆开所有水平unit，然后做分布式训练。或采用ForestLayer方法，由一个Actor来做KFold。
* [ ] 以进化学习的角度来做自适应的超参调优：随机按概率分布地选择一个摇臂，然后选择一个超参运行。
* [ ] gen_random_state设置问题，BaseEstimator和CompositeEstimator不一样
