import argparse
import yaml

class Flags:
    def __init__(self):
        '''
        Class for parsing command line arguments
        '''
        self.parser = argparse.ArgumentParser(description="MatPlat inputs")
        self.add_required_args()
        self.add_optional_args()
        args, unknown = self.get_args()
        default_config = self.parse_yml(args.config_file)
        updated_config = self.update_from_args(unknown, default_config)

        # add required args into updated config
        for key, val in vars(args).items():
            setattr(updated_config, key, val)
        self.updated_config = updated_config

    def add_required_args(self):
        '''
        Add the necessary command line arguments
        '''

        self.parser.add_argument(
            "--task_type",
            choices=["train", "test", "predict", "visualize", "hyperparameter", "CV", "matbench","matbenchTuning"],
            required=True,
            type=str,
            help="Type of task to perform: train, test, predict, visualize, hyperparameter",)
        self.parser.add_argument(
            "--config_file",
            required=True,
            type=str,
            help="Default parameters for training",
            default='./config.yml'
        )
    def  add_optional_args(self):
        self.parser.add_argument(
            "--matbenchTasklist",
            nargs='+',
            required=False,
            type=int,
            help="Indices for matbench tasks",
            default=[1]
        )
    def get_args(self):
        args, unknown = self.parser.parse_known_args()
        return args, unknown

    def parse_yml(self, yml_file):
        def recursive_flatten(nestedConfig):
            for k, v in nestedConfig.items():
                if isinstance(v, dict):
                    if k.split('_')[-1] == 'args':
                        flattenConfig[k] = v
                    else:
                        recursive_flatten(v)
                else:
                    flattenConfig[k] = v

        flattenConfig = {}
        with open(yml_file, 'r') as f:
            nestedConfig = yaml.load(f, Loader=yaml.FullLoader)
        recursive_flatten(nestedConfig)

        config = argparse.Namespace()
        for key, val in flattenConfig.items():
            setattr(config, key, val)
        return config
    
    def update_from_args(self, unknown_args, ymlConfig):
        assert len(unknown_args) % 2 == 0, f"Please Check Arguments, {' '.join(unknown_args)}"
        for key, val in zip(unknown_args[0::2], unknown_args[1::2]):
            key = key.strip("--")
            val = self.parse_value(val)
            setattr(ymlConfig, key, val)
        return ymlConfig

    def parse_value(self, value):
        import ast
        """
        Parse string as Python literal if possible and fallback to string.
        """
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Use as string if nothing else worked
            return value
