from das import ParameterSpace
from das.crossvalidate import cross_validate_score
from das.HypTuner.config_gen.config_space import ParamSpace2ConfigSpace
from das.performance_evaluation import eval_performance, score_to_loss
verbose = False
import das
import logging
logger = logging.getLogger(das.logger_name)


class Pipeline(object):
    """
    model: 列表类型，列表每一项为一个tuple，tuple的第一项为算法的名称，第二项为算法的实例
    例如：[("Normalizer", sklearn.normalizer()), ("PCA", sklearn.pca()), ("LogisticRegression", sklearn.LogisticRegression())]
    """
    def __init__(self,
                 model):

        self.model = model
        self.current_state = "start"
        self.reward = None

        # TODO: model要求除最后一项，前面的每一个算法实例都包含fit和transform方法，最后一行包括fit以及predict方法

    def fit(self, X, y=None):
        # logger.debug("pipeline fitting")
        len_model = len(self.model)
        # logger.debug("333333 len_model: {}".format(len_model))
        # logger.debug("pipeline ffitting")
        tmp_X = None
        for aa in range(len_model):
            self.current_state = self.model[aa][0]
            logger.debug("current_state: {}".format(self.current_state))
            if aa < len_model-1:

                if tmp_X is None:
                    self.model[aa][1].fit(X, y)
                    tmp_X = self.model[aa][1].transform(X)
                else:
                    self.model[aa][1].fit(tmp_X, y)
                    tmp_X = self.model[aa][1].transform(tmp_X)

                # tmp_X = self.model[aa][1].transform(X)
                # self.model[aa][1].fit(X, y)
                # X = self.model[aa][1].transform(X)
            else:
                if tmp_X is None:
                    self.model[aa][1].fit(X, y)
                else:
                    self.model[aa][1].fit(tmp_X, y)
                # self.model[aa][1].fit(X, y)

    def predict(self, X):

        len_model = len(self.model)
        y_hat = None
        tmp_X = None
        for aa in range(len_model):

            if aa < len_model - 1:
                if tmp_X is None:
                    tmp_X = self.model[aa][1].transform(X)
                else:
                    tmp_X = self.model[aa][1].transform(tmp_X)
            else:
                if tmp_X is None:
                    y_hat = self.model[aa][1].predict(X)
                else:
                    y_hat = self.model[aa][1].predict(tmp_X)

        return y_hat

    def predict_proba(self, X):

        len_model = len(self.model)
        y_hat = None
        tmp_X = None
        for aa in range(len_model):
            if aa < len_model - 1:
                if tmp_X is None:
                    tmp_X = self.model[aa][1].transform(X)
                else:
                    tmp_X = self.model[aa][1].transform(tmp_X)
            else:
                if tmp_X is None:
                    y_hat = self.model[aa][1].predict_proba(X)
                else:
                    y_hat = self.model[aa][1].predict_proba(tmp_X)
        return y_hat

    def get_params(self):
        params_dict = {}
        for m_name, m_instance in self.model:
            params = mapping_key(m_name, m_instance.get_params())
            params_dict.update(params)
        return params_dict

    def set_params(self, **params):
        for k in params.keys():
            k_pre, _ , k_post = k.partition("__")
            tmp_model = None
            for aa in self.model:
                if aa[0] == k_pre:
                    tmp_model = aa[1]
                    break
            # print(tmp_model.get_model_name(), tmp_model.get_params())
            # print(tmp_model.model.get_params())
            # print("k_post", k_post)
            if tmp_model is None or not hasattr(tmp_model.model, k_post):
                # print(k_post, tmp_model.print_model())
                raise Exception("non-valid parameters in BaseAlgorithm")
            setattr(tmp_model.model, k_post, params[k])
            setattr(tmp_model, k_post, params[k])

    def set_config(self, **params):
        for k in params.keys():
            k_pre, _, k_post = k.partition("__")
            tmp_model = None
            for aa in self.model:
                if aa[0] == k_pre:
                    tmp_model = aa[1]
                    break
            if tmp_model is None or not hasattr(tmp_model.model, k_post):
                # print(k_post, tmp_model.print_model())
                raise Exception("non-valid parameters")
            setattr(tmp_model.model, k_post, params[k])

    def get_configuration_space(self):
        # tps = self.model[0][1].get_configuration_space()
        tps = ParameterSpace.ParameterSpace()
        # 重新命名

        # for aa in tps.get_space():
        #     if not aa.get_name().startswith(self.model[0][0]+"__"):
        #         aa.set_name(self.model[0][0]+"__"+aa.get_name())
        #print("self.model", self.model)
        for aa in self.model:
            # if aa[0] == self.model[0][0]:
            #     continue
            # else:
            if True:

                tmp_space = aa[1].get_configuration_space().get_space()
                # print(tmp_space)
                for rr in tmp_space:
                    #if True: print("rr.get_name()", rr.get_name())
                    if not rr.get_name().startswith(aa[0]+"__"):
                        rr.set_name(aa[0]+"__"+rr.get_name())
                tps.merge(tmp_space)

                # tmp_relation = aa[1].get_parameter_relation()
                # for rr in tmp_relation:
                #     rr.set_name(aa[0]+"__"+rr.get_name())
                # for rr in tmp_relation:
                #     tps.add_parameter_relation(rr)

        return tps

    def new_estimator(self, config=None):
        self.set_params(**config)
        return Pipeline(model=self.model)


    def compute(self,
                config_id=None,
                config=None,
                budgets=None,
                X=None,
                y=None,
                X_val=None,
                y_val=None,
                evaluation_rule=None,
                working_directory=".",
                task='classification',
                **kwargs):
        model = self.new_estimator(config=config)
        assert evaluation_rule is not None, "Evaluation rule is None, please provide a valid rule!"
        if task == 'clustering':
            model.fit(X)
            # print("X.shape=", X.shape)
            y_hat = model.predict(X)
            # print("y_hat.shape", y_hat.shape)
            # print("set(y_hat): {}".format(set(y_hat)))
            val_score = eval_performance(rule=evaluation_rule,
                                         X=X,
                                         y_score=y_hat)
        elif X_val is not None:
            model.fit(X, y)
            y_hat = model.predict(X_val)
            val_score = eval_performance(rule=evaluation_rule,
                                         y_true=y_val,
                                         y_score=y_hat)
        else:
            cv_fold = kwargs['validation_strategy_args']
            assert (1 < cv_fold <= 10), "CV Fold should be: 1 < fold <= 10"
            val_score, _ = cross_validate_score(model, X, y, cv=cv_fold, evaluation_rule=evaluation_rule)
            # val_score = np.mean(val_results)
        # TODO(huqiu): restrict val_score to 0~1, or re-construct performance evaluation
        self.reward = {'loss': score_to_loss(evaluation_rule, val_score),
                       'info': {'val_{}'.format(evaluation_rule): val_score}}
        return self.reward

    def get_config_space(self):
        das_config_space = self.get_configuration_space()
        cs = ParamSpace2ConfigSpace(das_config_space)
        return cs

    def print_model(self):

        if self.model is None:
            print("None")
        tmp = []
        for i in self.model:
            tmp.append(i[0])

        print(tmp)
        return tmp

    def get_model_name(self):

        if self.model is None:
            print("None")
        tmp = []
        for i in self.model:
            tmp.append(i[0])

        return tmp

    def get_model(self):
        return self.model


    def get_current_state(self):
        return self.current_state


def mapping_key(m_name, m_params):
    new_params = {}
    for key in m_params:
        new_params["{}__{}".format(m_name, key)] = m_params[key]
    return new_params


if __name__ == "__main__":

    from das.BaseAlgorithm.Preprocessing.SKLearnPreprocessing import Normalizer
    from das.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering import PCA
    from das.BaseAlgorithm.classification.SKLearnBaseAlgorithm import LogisticRegression
    from das.HyperparameterOptimizer import RandomSelector
    mm = [("Normalizer", Normalizer.Normalizer()), ("PCA", PCA.PCA()), ("LogisticRegression", LogisticRegression.LogisticRegression())]
    p = Pipeline(mm)
    print(p.get_model_name())
    cp = p.get_configuration_space()
    rs = RandomSelector.RandomSelector(cp)
    cand_config = rs.get_random_config()
    print(cand_config)
    p.set_params(**cand_config)


