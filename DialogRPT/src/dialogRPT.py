from DialogRPT.src.model import OptionInfer, JointScorer, Scorer
import torch

def getIntegrated(path_ranker, path_generator, cuda):
    opt = OptionInfer()
    ranker = JointScorer(opt = opt)
    ranker.load(path_ranker)
    from DialogRPT.src.generation import GPT2Generator, Integrated
    generator = GPT2Generator(path_generator, cuda)
    generator.predict = generator.predict_beam
    return Integrated(generator, ranker)
    