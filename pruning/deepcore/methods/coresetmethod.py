class CoresetMethod(object):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, num_tasks=None, device=None, **kwargs):
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")
        self.dst_train = dst_train
        self.num_classes = num_tasks
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args
        self.print_freq = 50
        self.device = device

        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)

    def select(self, **kwargs):
        return

