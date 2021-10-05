from torch_snippets import P, os
from torch_snippets.registry import Config, AttrDict, registry

class Task:
    def __init__(self, config):
        self.get_config(config)
        self.parse_config()

    def get_config(self, config):
        if config is None:
            config = os.environ['CONFIG']
        self.config = Config().from_disk(config)

    def parse_config(self):
       self.config = AttrDict(registry.resolve(self.config))

    def download_data(self):
        import torch_snippets
        from fastdownload import FastDownload
        source = self.config.project.data_source_url
        d = FastDownload(
            base=P(self.config.training.dir).parent,
            data='./',
            module=torch_snippets)
        d.get(source)
