from binary_similarity.gemini import run_gemini_model

from dataset_creation.FunctionAnalyzerRadare import RadareFunctionAnalyzer
from binary_similarity.gemini import run_gemini_model

import numpy as np
import os
from tqdm import tqdm

from multiprocessing import Pool
from collections import ChainMap

import pickle


def alpha_function(f_emb, target_embeddings):
    threshold = 0.95

    for el in target_embeddings:
        dot_product = np.dot(f_emb, target_embeddings[el])
        if dot_product >= threshold:
            return 1
    return 0


processes = 16


def run_gemini_parallel(parameters):

    path = parameters[0]
    name = parameters[1]

    analyzer = RadareFunctionAnalyzer(os.path.join(path, name), use_symbol=False)
    functions = analyzer.analyze()
    return functions


to_analyze = ["Winnti Group"]
analyzed = ["APT28", "APT30", "Hurricane Panda", "Lazarus Group", "Mirage", "Sandwork", "Shiqiang",
            "Transparent Tribe", "Violin Panda", "Winnti Group", "Patchwork", "APT29", "Carbanak"]


def analyze_apts():
    folder = "/data/brick1/advBinarySimilarity/geminiAPT/apt_groups/"
    # folder = "/media/gianluca/Data/Corsi/AI4CTIA/apt_groups/"

    for path, subdirs, files in os.walk(folder):

        apt_fam = path.rsplit("/", 1)[1]
        print(apt_fam)

        parameters = []
        if apt_fam in to_analyze:
            for name in files:
                parameters.append([path, name])
            with Pool(processes) as p:
                results = p.map(run_gemini_parallel, parameters)
                p.close()
                p.join()

            print("++++ START GEMINI")
            embs = {}
            emb_list = []
            if results:
                for functions in tqdm(results):
                    if functions:
                        emb_list.append(run_gemini_model(functions))
                for idx in range(len(emb_list)):
                    embs[files[idx]] = {}
                    embs[files[idx]]['functions'] = emb_list[idx]
                    embs[files[idx]]['apt'] = apt_fam

            # folder_embs = dict(ChainMap(*results))

            with open('/data/brick1/advBinarySimilarity/geminiAPT/embs/{}.pickle'.format(apt_fam), 'wb') as handle:
                pickle.dump(embs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(apt_fam)


def group_in_single_file(folder, filename):

    embeddings = {}
    for _file in tqdm(os.listdir(folder)):
        if _file.endswith(".pickle"):
            myfile = open(folder+_file, "rb")
            curr_db = pickle.load(myfile)
            if curr_db:
                for el in curr_db:
                    embeddings[el] = curr_db[el]
            myfile.close()

    myfile = open(filename, "wb")
    pickle.dump(embeddings, myfile)
    myfile.close()
    print("OK")


def extract_embs(filename):

    f = open(filename, "rb")
    data = pickle.load(f)
    f.close()

    apt = []
    file = []
    embeddings = []
    for k in data.keys():
        # exlude functions with less than 3 nodes
        emb = [f['embedding'] for f in data[k]['functions'] if f['num_nodes'] > 0]
        emb = np.asarray(emb)
        if emb.shape[0] > 0:  # and data[k]['apt'] != 'Patchwork':
            embeddings.append(emb)
            apt.append(data[k]['apt'].replace("\n", ""))
            file.append(k)

    print("OK")


if __name__ == "__main__":
    #analyze_apts()

    group_in_single_file("/data/brick1/advBinarySimilarity/geminiAPT/embs/",
                         "/data/brick1/advBinarySimilarity/geminiAPT/gemini_embs.pickle")

    # extract_embs("/data/brick1/advBinarySimilarity/geminiAPT/gemini_embs.pickle")

    """
    path = ""

    apt_exec_1 = "/media/gianluca/Data/Corsi/AI4CTIA/apt_groups/Transparent Tribe/2463d1ff1166e845e52a0c580fd3cb7d"
    apt_exec_2 = "/media/gianluca/Data/Corsi/AI4CTIA/apt/dbd7d010c4657b94f49ca85e4ff88790"

    analyzer_1 = RadareFunctionAnalyzer(apt_exec_1, use_symbol=False)
    functions_1 = analyzer_1.analyze()

    functions_embds_1 = run_gemini_model(functions_1)

    analyzer_2 = RadareFunctionAnalyzer(apt_exec_2, use_symbol=False)
    functions_2 = analyzer_2.analyze()

    functions_embds_2 = run_gemini_model(functions_2)

    alphas = 0
    for el in functions_embds_1:
        alphas += alpha_function(functions_embds_1[el], functions_embds_2)

    sim = alphas/max(len(functions_embds_1), len(functions_embds_2))

    print("SIM: {}".format(sim))
    """

