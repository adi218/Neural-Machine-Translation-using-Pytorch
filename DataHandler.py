import os
import pandas as pd


class DataHandler:

    def load_train(self, folder="./data/", rows=-1):
        for file in os.listdir(folder):
            file_path = os.path.join(os.path.abspath(folder), file)
            if file_path.__contains__("train"):
                if file_path.endswith("en"):
                    file_en = open(file_path)
                    dataset_en = self._read_file(file_en)
                elif file_path.endswith("vi"):
                    file_vi = open(file_path)
                    dataset_vi = self._read_file(file_vi)
        if rows != -1:
            return dataset_en.sample(rows), dataset_vi.sample(rows)
        return dataset_en, dataset_vi

    @staticmethod
    def _read_file(file):

        lines = file.readlines()
        lst_lines = [x.strip() for x in lines]
        return pd.DataFrame(lst_lines)

    def tokenizer(self):

