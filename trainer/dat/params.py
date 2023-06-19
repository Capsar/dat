from datetime import datetime


class Params:
    def __init__(self, epochs: int,
                 batch_size: int,
                 sparsity_budget: int,
                 learning_rate: float,
                 p_min: float,
                 use_cifarext: bool,
                 use_jointspar: bool):
        self.epochs = epochs
        self.batch_size = batch_size
        self.sparsity_budget = sparsity_budget
        self.p_min = p_min / self.sparsity_budget
        self.learning_rate = learning_rate
        self.use_cifarext = use_cifarext
        self.use_jointspar = use_jointspar

    def __str__(self):
        return f'**Params:**' \
               f'\n\tEpochs: {self.epochs}' \
               f'\n\tSparsity budget: {self.sparsity_budget}' \
               f'\n\tp min: {self.p_min}' \
               f'\n\tLearning rate: {self.learning_rate}' \
               f'\n\tDataset: {"CifarExt" if self.use_cifarext else "Cifar10"}' \
               f'\n\tUsing JointSpar: {self.use_jointspar}'

    def get_filename(self) -> str:
        jointspar_suffix = '_jointspar' if self.use_jointspar else ''
        cifar_suffix = '_cifarext' if self.use_cifarext else ''
        today_str = datetime.now().strftime('%HH%MM-%d%m%Y')
        return f'./runs/{today_str}' \
               f'_{self.epochs}epochs' \
               f'_{self.batch_size}batch' \
               f'_{self.sparsity_budget}sparsity' \
               f'_{self.p_min}pmin' \
               f'_{self.learning_rate}lr' \
               f'{jointspar_suffix}{cifar_suffix}.obj'


# params1 = Params(
#     epochs=100,
#     batch_size=2048,
#     sparsity_budget=30,
#     learning_rate=0.01,
#     use_cifarext=True,
#     use_jointspar=True
# )
#
# print(params1.get_filename())