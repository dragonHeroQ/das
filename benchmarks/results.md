
### RS LETTER 3600-240
Best Config: {'b1_algo': 'BernoulliNB', 'b1_num': 1, 'b2_algo': 'ExtraTreesClassifier', 'b2_num': 2}, best nLayer: 1
Final Train Score=0.9480625, Test Score=0.96475

### RS DIGITS 3600-240
[2019-01-21 14:46:54,989 RandomSearch.gen_best_record] Best Config: {'b1_algo': 'ExtraTreesClassifier', 'b1_num': 4, 'b2_algo': 'KNeighborsClassifier', 'b2_num': 4}, best nLayer: 1
Final Train Score=0.9758935993349959, Test Score=0.9865319865319865

### BO slave023_ucb_adult_r0_time_18000_nbandit_80_nstage_5_TL_700_01231436
[2019-01-23 19:36:19,101 SMBO.fit] Random Search fit TimeCost = 17998.14774441719
[2019-01-23 19:36:19,101 SMBO.gen_best_record] Best Config: {'b1_algo': 'LGBClassifier', 'b1_num': 3, 'b2_algo': 'AdaboostClassifier', 'b2_num': 3}, best nLayer: 1
[2019-01-23 19:36:36,443 BaseEvaluator.fit_predict] TimeCost: 17.339648008346558, Exceptions: None
Final Train Score=0.8713491600380824, Test Score=0.873042196425281
[2019-01-23 19:36:36,461 BaseEvaluator.save_learning_curve] Learning Curve Saved in ./lcvs/slave023_ucb_adult_r0_time_18000_nbandit_80_nstage_5_TL_700_01231436.lcv

### AdaUCB slave025_ucb_adult_r0_time_18000_nbandit_80_nstage_5_TL_700_01231436
...
REJECTING: tmp_success_Rate = {69: 0.0, 15: 1.0}
Newly REJECTED 1/79, LEAVE 1
Round 4, from 2 bandits to 1 bandits
best ind: 11
best val loss:  0.14072049384232665
best bandit: DeepArchiClassifier([ DecisionTreeClassifier x 3, DecisionTreeClassifier x 4 ])
best parameter: {'0/0/block0/0#0/N/DecisionTreeClassifier/criterion': 'entropy', '0/0/block0/0#0/N/DecisionTreeClassifier/splitter': 'best', '0/0/block0/0#0/N/DecisionTreeClassifier/min_samples_split': 2, '0/0/block0/0#0/N/DecisionTreeClassifier/min_samples_leaf': 2, '0/0/block0/0#0/N/DecisionTreeClassifier/max_features': 'auto', '0/0/block0/0#0/N/DecisionTreeClassifier/presort': False, '0/1/block1/0#1/N/DecisionTreeClassifier/criterion': 'entropy', '0/1/block1/0#1/N/DecisionTreeClassifier/splitter': 'random', '0/1/block1/0#1/N/DecisionTreeClassifier/min_samples_split': 2, '0/1/block1/0#1/N/DecisionTreeClassifier/min_samples_leaf': 4, '0/1/block1/0#1/N/DecisionTreeClassifier/max_features': None, '0/1/block1/0#1/N/DecisionTreeClassifier/presort': True}
[2019-01-23 19:36:16,709 DeepArchiClassifier.fit_predict] Layer 1: train_accuracy_score = 0.8592795061576733
[2019-01-23 19:36:21,541 DeepArchiClassifier.fit_predict] Layer 2: train_accuracy_score = 0.8471177175148183
[2019-01-23 19:36:25,939 DeepArchiClassifier.fit_predict] Layer 3: train_accuracy_score = 0.8449371948035994
[2019-01-23 19:36:30,410 DeepArchiClassifier.fit_predict] Layer 4: train_accuracy_score = 0.8418660360554037
[2019-01-23 19:36:34,908 DeepArchiClassifier.fit_predict] Layer 5: train_accuracy_score = 0.8424188446300789
[2019-01-23 19:36:34,972 BaseEvaluator.fit_predict] TimeCost: 21.885574340820312, Exceptions: None
Final Train Score=0.8592795061576733, Test Score=0.862600577360113
[2019-01-23 19:36:34,993 BaseEvaluator.save_learning_curve] Learning Curve Saved in ./lcvs/slave025_ucb_adult_r0_time_18000_nbandit_80_nstage_5_TL_700_01231436.lcv

### HB3 eta=3 slave025_hb3.py_adult_r0_time_18000_nbandit_80_nstage_5_TL_700_01232130.lcv
best ind: 15
best val loss:  0.1370965265194558
best bandit: DeepArchiClassifier([ GBDTClassifier x 4, ExtraTreesClassifier x 2 ])
best parameter: {'0/0/block0/0#0/N/GBDTClassifier/loss': 'deviance',
 '0/0/block0/0#0/N/GBDTClassifier/learning_rate': 0.10855709856182173,
  '0/0/block0/0#0/N/GBDTClassifier/n_estimators': 56,
   '0/0/block0/0#0/N/GBDTClassifier/criterion': 'friedman_mse',
    '0/0/block0/0#0/N/GBDTClassifier/min_samples_leaf': 1,
     '0/0/block0/0#0/N/GBDTClassifier/min_samples_split': 3,
      '0/0/block0/0#0/N/GBDTClassifier/max_features': 'log2',
       '0/1/block1/0#1/N/ExtraTreesClassifier/n_estimators': 156,
        '0/1/block1/0#1/N/ExtraTreesClassifier/criterion': 'gini',
         '0/1/block1/0#1/N/ExtraTreesClassifier/min_samples_split': 16,
          '0/1/block1/0#1/N/ExtraTreesClassifier/min_samples_leaf': 7,
           '0/1/block1/0#1/N/ExtraTreesClassifier/max_features': None,
            '0/1/block1/0#1/N/ExtraTreesClassifier/bootstrap': True,
             '0/1/block1/0#1/N/ExtraTreesClassifier/oob_score': False}
[2019-01-24 02:30:46,530 DeepArchiClassifier.fit_predict] Layer 1: train_accuracy_score = 0.8577439267835755
[2019-01-24 02:31:07,963 DeepArchiClassifier.fit_predict] Layer 2: train_accuracy_score = 0.8629034734805442
[2019-01-24 02:31:29,482 DeepArchiClassifier.fit_predict] Layer 3: train_accuracy_score = 0.8621049722060133
[2019-01-24 02:31:50,989 DeepArchiClassifier.fit_predict] Layer 4: train_accuracy_score = 0.8625349344307607
[2019-01-24 02:32:12,693 DeepArchiClassifier.fit_predict] Layer 5: train_accuracy_score = 0.8603237001320598
[2019-01-24 02:32:34,899 DeepArchiClassifier.fit_predict] Layer 6: train_accuracy_score = 0.8603851233070238
[2019-01-24 02:32:34,966 BaseEvaluator.fit_predict] TimeCost: 127.06627130508423, Exceptions: None
Final Train Score=0.8629034734805442, Test Score=0.8645660585959093
[2019-01-24 02:32:34,985 BaseEvaluator.save_learning_curve] Learning Curve Saved in ./lcvs/slave025_hb3.py_adult_r0_time_18000_nbandit_80_nstage_5_TL_700_01232130.lcv

### RS slave023_rs.py_adult_r0_time_18000_nbandit_80_nstage_5_TL_700_01232130.lcv
[2019-01-24 02:30:40,725 SMBO.fit] Random Search fit TimeCost = 17998.07194375992
[2019-01-24 02:30:40,726 SMBO.gen_best_record] Best Config: {'b1_algo': 'LGBClassifier', 'b1_num': 2, 'b2_algo': 'AdaboostClassifier', 'b2_num': 4}
[2019-01-24 02:30:40,726 SMBO.gen_best_record] Best Reward: {'loss': 0.12923436012407485, 'val_accuracy_score': 0.8707656398759251, 'best_nLayer': 1, 'time_cost': 163.43459725379944, 'exception': None}
[2019-01-24 02:31:03,398 BaseEvaluator.fit_predict] TimeCost: 22.669684410095215, Exceptions: None
Final Train Score=0.8707656398759251, Test Score=0.8727965112708065
[2019-01-24 02:31:03,417 BaseEvaluator.save_learning_curve] Learning Curve Saved in ./lcvs/slave023_rs.py_adult_r0_time_18000_nbandit_80_nstage_5_TL_700_01232130.lcv



