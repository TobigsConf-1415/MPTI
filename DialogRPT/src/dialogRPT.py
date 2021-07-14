from DialogRPT.src.model import OptionInfer, JointScorer, Scorer
import torch

def getIntegrated(path_ranker, path_generator, cpu):
    opt = OptionInfer()
    ranker = JointScorer(opt = opt)
    ranker.load(path_ranker)
    from DialogRPT.src.generation import GPT2Generator, Integrated
    cuda = False if cpu else torch.cuda.is_available()
    generator = GPT2Generator(path_generator, cuda)
    generator.predict = generator.predict_beam
    return Integrated(generator, ranker)
    