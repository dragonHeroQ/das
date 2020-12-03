from sklearn.externals import joblib
import os
import das
import logging
logger = logging.getLogger(das.logger_name)


def _check_model_folder(model_folder):
    # verify whether the model_folder exists
    if not os.path.exists(model_folder):
        logger.error("the folder: {} does not exist!".format(model_folder))
        raise Exception("the folder: {} does not exist!".format(model_folder))


def _check_model_file(model_folder, version, ind):
    # verify whether the model_file exists
    if not os.path.exists(model_folder + "/model_{}_{}.pkl".format(version, ind)):
        logger.error("the model file: {} does not exist!".format("model_{}_{}.pkl".format(version, ind)))
        raise Exception("the model file: {} does not exist!".format("model_{}_{}.pkl".format(version, ind)))


def has_model_file(model_folder, version, ind):
    if not os.path.exists(model_folder + "/model_{}_{}.pkl".format(version, ind)):
        return False
    else:
        return True


def _construct_model_file_name(model_folder, version, ind):
    return model_folder + "/model_{}_{}.pkl".format(version, ind)


def load_model(model_folder, version, ind):
    # load model from file
    _check_model_folder(model_folder)
    _check_model_file(model_folder, version, ind)
    tmp_model = joblib.load(model_folder + "/model_{}_{}.pkl".format(version, ind))
    return tmp_model


def save_model(model, model_folder, version, ind, compress_level=3) :
    # save model to file
    _check_model_folder(model_folder=model_folder)
    joblib.dump(model, _construct_model_file_name(model_folder, version, ind), compress=compress_level)
