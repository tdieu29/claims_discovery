import time

import faiss

from colbert.indexing.faiss_index_gpu import FaissIndexGPU
from config.config import logger


class FaissIndex:
    def __init__(self, dim, partitions, nprobe):
        self.dim = dim
        self.partitions = partitions
        self.nprobe = nprobe

        self.gpu = FaissIndexGPU()
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)  # faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)
        assert not index.is_trained

        return quantizer, index

    def train(self, train_data):
        logger.info(f"#> Training now (using {self.gpu.ngpu} GPUs)...")

        if self.gpu.ngpu > 0:
            self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        self.index.train(train_data)
        logger.info(f"Index training time: {time.time() - s}")

        if self.gpu.ngpu > 0:
            self.gpu.training_finalize()

        assert self.index.is_trained

    def add(self, data):
        logger.info(f"Add data with shape {data.shape} (offset = {self.offset})..")

        if self.gpu.ngpu > 0 and self.offset == 0:
            self.gpu.adding_initialize(self.index)

        if self.gpu.ngpu > 0:
            self.gpu.add(self.index, data, self.offset)
        else:
            self.index.add(data)

        self.offset += data.shape[0]

    def save(self, output_path):
        logger.info(f"Writing index to {output_path} ...")

        self.index.nprobe = self.nprobe
        faiss.write_index(self.index, output_path)
