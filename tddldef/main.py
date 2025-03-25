from pathlib import Path
from typing import List, Optional
import datetime

from utils.db_manager import DbManager
from utils.utils import get_HH_mm_ss
from src.cases_generation.model_info_generator import ModelInfoGenerator
from src.cases_generation.model_generator import ModelGenerator
from src.cases_generation.data_generator import DataGenerator
from src.incons_detection.trainer import Trainer
from src.incons_detection.comparator import Comparator
from utils.selection import Selector


class TrainingDebugger(object):

    def __init__(self, config: dict, generate_mode: str = 'template', timeout: float = 300):
        super().__init__()
        self.__output_dir = config['output_dir']
        self.__db_manager = DbManager(config['db_path'])
        # Selector 处理器
        from utils.utils import layer_types, layer_conditions
        self.__selector = Selector(layer_types=layer_types,
                                   layer_conditions=layer_conditions)
        self.__model_info_generator = ModelInfoGenerator(config['model'], self.__db_manager, self.__selector, generate_mode)
        self.__model_generator = ModelGenerator(config['model']['var']['weight_value_range'], config['backends'], self.__db_manager, self.__selector, timeout)
        self.__training_data_generator = DataGenerator(config['training_data'])
        self.__weights_trainer = Trainer(self.__db_manager, timeout)
        self.__weights_comparator = Comparator(self.__db_manager)


    def run_generation_for_dataset(self, dataset_name: str):
        # 随机生成model
        print('model生成开始...')
        json_path, _, _, model_id, exp_dir = self.__model_info_generator.generate_for_dataset(save_dir=self.__output_dir, dataset_name=dataset_name)
        ok_backends = self.__model_generator.generate(json_path=json_path,
                                                      model_id=model_id,
                                                      exp_dir=exp_dir)
        print(f'model生成完毕: model_id={model_id} ok_backends={ok_backends}')

        if len(ok_backends) >= 2:  # 否则没有继续实验的必要
            # 复制数据集
            print('training data生成开始...')
            import shutil
            shutil.copytree(str(Path('dataset') / dataset_name), str(Path(exp_dir) / 'dataset'))
            print('training data生成完毕.')

        return model_id, exp_dir, ok_backends

    def run_detection(self, model_id: int, exp_dir: str, ok_backends: List[str],  loss: Optional[str] = None, optimizer: Optional[str] = None):
        if len(ok_backends) >= 2:
            print('Training开始...')
            status, backends_outputs, backends_losses, backends_loss_grads, backends_grads, ok_backends = self.__weights_trainer.train(model_id=model_id,
                                                                                                                                       exp_dir=exp_dir,
                                                                                                                                       ok_backends=ok_backends,
                                                                                                                                       loss=loss,
                                                                                                                                       optimizer=optimizer)
            print(f'Training结束: ok_backends={ok_backends}')

            self.__db_manager.record_status(model_id, status)

        if len(ok_backends) >= 2:
            print('Compare开始...')
            self.__weights_comparator.compare(model_id=model_id,
                                              exp_dir=exp_dir,
                                              backends_outputs=backends_outputs,
                                              backends_losses=backends_losses,
                                              backends_loss_grads=backends_loss_grads,
                                              backends_grads=backends_grads,
                                              # backends_weights=backends_weights,
                                              ok_backends=ok_backends)
            print('Compare结束.')

        return ok_backends

    def get_coverage(self):
        return self.__selector.coverage()


def main(dataset_name: str, case_num: int, generate_mode: str):
    config = {
        'model': {
            'var': {
                'tensor_dimension_range': (2, 5),
                'tensor_element_size_range': (2, 5),
                'weight_value_range': (-10.0, 10.0),
                'small_value_range': (0, 1),
                'vocabulary_size': 1001,
            },
            'node_num_range': (5, 5),
            'dag_io_num_range': (1, 3),
            'dag_max_branch_num': 2,
            'cell_num': 3,
            'node_num_per_normal_cell': 10,
            'node_num_per_reduction_cell': 2,
        },
        'training_data': {
            'instance_num': 10,
            'element_val_range': (0, 100),
        },
        'db_path': str(Path.cwd() / 'data' / f'{dataset_name}.db'),
        'output_dir': str(Path.cwd() / 'data' / f'{dataset_name}_output'),
        'report_dir': str(Path.cwd() / 'data' / f'{dataset_name}_report'),
        'backends': ['tensorflow', 'cntk'],
        'distance_threshold': 0,
    }
    debugger = TrainingDebugger(config, generate_mode, 300)

    start_time = datetime.datetime.now()
    # 现成数据集 + 随机模型
    for i in range(case_num):
        print(f"********** Model {i} **********")
        try:
            print(f"------------- Generating model {i}  -------------")
            model_id, exp_dir, ok_backends = debugger.run_generation_for_dataset(dataset_name)
            print(f"------------- Detecting Model {i}  -------------")
            ok_backends = debugger.run_detection(model_id, exp_dir, ok_backends)
        except Exception:
            import traceback
            traceback.print_exc()

    end_time = datetime.datetime.now()
    time_delta = end_time - start_time
    h, m, s = get_HH_mm_ss(time_delta)
    print(f"All is done: Time used: {h} hour,{m} min,{s} sec")
    percent, _ = debugger.get_coverage()
    print(f"All model coverage {percent} %")


if __name__ == '__main__':
    DATASET_NAME = 'mnist' # `cifar10`, `mnist`, `sinewave`
    CASE_NUM = 50
    GENERATE_MODE = 'seq'  # `seq`, `merging`, `dag`

    main(DATASET_NAME, CASE_NUM, GENERATE_MODE)

